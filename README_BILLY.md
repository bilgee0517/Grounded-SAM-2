# Unsupervised action classification

## Grounded SAM 2 

### Step-by-Step Explanation
1. Environment Setup and Model Initialization:

- The script starts by setting up the environment for running SAM 2 using PyTorch and enabling certain CUDA optimizations if the hardware supports it.
- It then initializes the SAM 2 video predictor and the SAM 2 image predictor using model configuration and checkpoint files.
- The script also loads the Grounding DINO model from Huggingface's Transformers library, which will be used for zero-shot object detection based on a text prompt.
2. Frame Extraction:

- The script loads the video specified by args.video_path and extracts all frames, saving them as images in a directory specified by args.source_video_frame_dir.
- These frames will later be used for object detection and segmentation.
3. Cosine Similarity Frame Sampling:

- To avoid redundant processing of similar frames, the script calculates the cosine similarity between consecutive frames.
- Frames with similarity below a certain threshold are selected for further processing.
4. Object Detection and Mask Prediction:

- For each selected frame, the script uses the Grounding DINO model to detect objects based on a provided text prompt (args.text_prompt).
- It then uses the SAM 2 image predictor to generate segmentation masks for the detected objects.
5. Prompting SAM 2:

- Depending on the specified prompt type (point, box, or mask), the script registers the detected objects and their corresponding prompts with the SAM 2 video predictor.
- This step involves adding new points, boxes, or masks to the video predictor’s inference state.
6. Segmentation Propagation:

- The script propagates the segmentation masks across the entire video. This results in a set of segmented masks for each frame.
7. Visualization and Saving:

- For each frame, the script combines the masks of all detected objects into a single mask and applies it to the frame to remove the background.
- The processed frames are then saved in the directory specified by args.save_tracking_results_dir.
8. Video Creation:

- Finally, the script converts the sequence of processed frames back into a video and saves it to args.output_video_path.

## Example Usage

To set up Grounded SAM 2, follow the installation instruction in the readme file in Grounded-SAM-2 folder. After setting up Grounded SAM 2, you can download the kinetics5per dataset by running 


```bash
cd data
python download_data.py
```

Or after directly importing your video, you can run 


```bash
cd data
python grounded_sam.py  --video_path "path/to/your/video.mp4" 
                      --text_prompt "your text prompt" 
                      --output_video_path "path/to/output/video.mp4" 
                      --source_video_frame_dir "directory/to/save/frames" 
                      --save_tracking_results_dir "directory/to/save/results" 
```


## Creating Embeddings and Clustering

After processing the videos for embedding, you can follow the process in Processing.ipynb . You can freely change the encoder model, pooling strategies, and number of frames to sample as you want for better accuracy. 

In the current design choices, it is using Kmeans for clustering, and maxpooling. Also, there are 8 frames sampled for small and large step sizes. Large step sizes are adjusted based on the total number of frames but small step size is not adjusted. The reason is based on assumption that large steps will get the temporal features and small steps will get spatial features. 