import os
import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import argparse
from sklearn.metrics.pairwise import cosine_similarity


# Function to compute cosine similarity between two frames
def compute_cosine_similarity(frame1, frame2):
    frame1 = frame1.astype(np.float32).flatten()
    frame2 = frame2.astype(np.float32).flatten()
    frame1 = frame1 / np.linalg.norm(frame1)
    frame2 = frame2 / np.linalg.norm(frame2)
    return np.dot(frame1, frame2)

def sample_frames_with_cosine_similarity(frames, similarity_threshold=0.9):
    selected_frames = [0]  # Start with the first frame
    last_selected_frame = frames[0]

    for i in range(1, len(frames)):
        current_frame = frames[i]
        similarity = compute_cosine_similarity(last_selected_frame, current_frame)
        if similarity < similarity_threshold:
            selected_frames.append(i)
            last_selected_frame = current_frame

    return selected_frames

def main(args):
    """
    Step 1: Environment settings and model initialization for SAM 2
    """
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    video_predictor = build_sam2_video_predictor(args.model_cfg, args.sam2_checkpoint)
    sam2_image_model = build_sam2(args.model_cfg, args.sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # build grounding dino from huggingface
    processor = AutoProcessor.from_pretrained(args.model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    """
    Custom video input directly using video files
    """
    video_info = sv.VideoInfo.from_video_path(args.video_path)  # get video info
    print(video_info)
    frame_generator = sv.get_video_frames_generator(args.video_path, stride=1, start=0, end=None)

    # saving video to frames
    source_frames = Path(args.source_video_frame_dir)
    source_frames.mkdir(parents=True, exist_ok=True)

    all_frames = []
    with sv.ImageSink(
        target_dir_path=source_frames, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            all_frames.append(frame)
            sink.save_image(frame)

    # Use cosine similarity to select frames for annotation
    selected_frame_indices = sample_frames_with_cosine_similarity(all_frames, similarity_threshold=0.9)
    ann_frame_idx_list = [selected_frame_indices[i] for i in range(0, len(selected_frame_indices), max(len(selected_frame_indices) // len(all_frames), 1))]

    print(f"Number of selected frames to use groundingdino {len(ann_frame_idx_list)}")
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(args.source_video_frame_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=args.source_video_frame_dir)

    """
    Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
    """

    for ann_frame_idx in ann_frame_idx_list:
        try:
            # prompt grounding dino to get the box coordinates on specific frame
            img_path = os.path.join(args.source_video_frame_dir, frame_names[ann_frame_idx])
            image = Image.open(img_path)
            inputs = processor(images=image, text=args.text_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            input_boxes = results[0]["boxes"].cpu().numpy()
            confidences = results[0]["scores"].cpu().numpy().tolist()
            class_names = results[0]["labels"]

            print(input_boxes)

            # prompt SAM image predictor to get the mask for the object
            image_predictor.set_image(np.array(image.convert("RGB")))

            # process the detection results
            OBJECTS = class_names

            print(OBJECTS)

            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # convert the mask shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

        except Exception as e:
            print(f"Error processing frame {ann_frame_idx}: {e}")
            continue


        """
        Step 3: Register each object's positive points to video predictor with separate add_new_points call
        """

        assert args.prompt_type_for_video in ["point", "box", "mask"], "SAM 2 video predictor only supports point/box/mask prompt"

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if args.prompt_type_for_video == "point":
            # sample the positive points from mask for each object
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        # Using box prompt
        elif args.prompt_type_for_video == "box":
            for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        # Using mask prompt is a more straightforward way
        elif args.prompt_type_for_video == "mask":
            for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    """
    Step 5: Visualize the segment results across the video and save them
    """

    if not os.path.exists(args.save_tracking_results_dir):
        os.makedirs(args.save_tracking_results_dir)
    
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(args.source_video_frame_dir, frame_names[frame_idx]))
        
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        
        # Combine all masks into a single mask for the entire frame
        combined_mask = np.sum(masks, axis=0).astype(np.uint8)
        
        # Apply the combined mask to the image to remove the background
        img_background_removed = cv2.bitwise_and(img, img, mask=combined_mask)
        
        # Save the frame with background removed
        cv2.imwrite(os.path.join(args.save_tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"), img_background_removed)

    """
    Step 6: Convert the annotated frames to video
    """

    create_video_from_images(args.save_tracking_results_dir, args.output_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos for segmentation.")
    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-tiny", help="Huggingface model ID")
    parser.add_argument("--video_path", type=str, default="../data/kinetics600_5per/kinetics600_5per/train/brushing teeth/BSJIVf8QeOI.mp4", help="Path to the video file")
    parser.add_argument("--text_prompt", type=str, default="humans. toothbrush .", help="Text prompt for object detection")
    parser.add_argument("--output_video_path", type=str, default="../data/sample.mp4", help="Path to save output video")
    parser.add_argument("--source_video_frame_dir", type=str, default="./custom_video_frames", help="Directory to save video frames")
    parser.add_argument("--save_tracking_results_dir", type=str, default="./tracking_results", help="Directory to save tracking results")
    parser.add_argument("--prompt_type_for_video", type=str, choices=["point", "box", "mask"], default="box", help="Prompt type for video (point, box, mask)")
    parser.add_argument("--sam2_checkpoint", type=str, default="./checkpoints/sam2_hiera_large.pt", help="Path to the SAM2 checkpoint file.")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml", help="Path to the SAM2 model config file.")
    
    args = parser.parse_args()
    
    main(args)
