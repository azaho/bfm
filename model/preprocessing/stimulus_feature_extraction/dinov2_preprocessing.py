import os
import json
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(description="Extract DINOv2 features from movie frames.")
    parser.add_argument(
        "--frame_interval",
        type=float,
        default=None,
        help="Time interval (in seconds) between consecutive frames to extract. "
             "If not specified, extracts every frame."
    )
    parser.add_argument(
        "--movie_path",
        type=str,
        required=True,
        help="Path to the movie file to process."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where to save the resulting file (should end with .npz)."
    )
    return parser.parse_args()

def get_dinov2_features(model, processor, image, device):
    """
    Extract DINOv2 features from an image.
    
    Args:
        model: DINOv2 model
        processor: DINOv2 image processor
        image: PIL Image
        device: torch device
    
    Returns:
        torch.Tensor: DINOv2 features
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the last hidden state (CLS token) as features
        features = outputs.last_hidden_state[:, 0, :]  # Shape: (1, hidden_size)
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu()

if __name__ == "__main__":
    args = parse_args()
    frame_interval = args.frame_interval
    movie_path = args.movie_path
    output_path = args.output_path

    # Validate input file exists
    if not os.path.exists(movie_path):
        raise FileNotFoundError(f"Movie file not found: {movie_path}")

    # Validate output path ends with .npz
    if not output_path.endswith('.npz'):
        output_path = output_path + '.npz'

    print("Loading DINOv2 model...")
    device = torch.device('cuda')
    model_name = "facebook/dinov2-base"  # You can change to dinov2-large or dinov2-small
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    print(f"Processing movie: {movie_path}")
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    features_list = []
    timestamps = []

    if frame_interval is None:
        # Extract every frame
        frame_indices = range(total_frames)
    else:
        # Extract frames at specified time intervals
        # Compute timestamps from 0 to duration (exclusive) with step frame_interval
        time_stamps = np.arange(0, duration, frame_interval)
        frame_indices = [int(ts * fps) for ts in time_stamps]

    for idx in tqdm(frame_indices, desc=f"Frames in {os.path.basename(movie_path)}"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If we can't read the frame, skip it
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        features = get_dinov2_features(model, processor, pil_image, device)
        features_list.append(features.squeeze(0).numpy())
        timestamps.append(idx / fps)

    cap.release()
    print(f"Extracted frames for {movie_path}")

    if features_list:
        features_array = np.stack(features_list)
        timestamps_array = np.array(timestamps)
        
        print(features_array.shape)
        print(timestamps_array.shape)
        
        # Save both features and timestamps in a single npz file
        np.savez(output_path, 
                 features=features_array, 
                 timestamps=timestamps_array)
        print(f"Saved features and timestamps to {output_path}")
    else:
        print(f"No frames extracted for {movie_path}")


##### TO RUN ######
# python dinov2_preprocessing.py --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --frame_interval 600 --output_path /om2/data/public/braintreebank_movies_dinov2_preprocessed/ant-man_dinov2_features.npz
##################### 
