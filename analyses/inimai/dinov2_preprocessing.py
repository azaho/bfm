import os
import json
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
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
    return parser.parse_args()

MOVIES_DIR = '/om2/data/public/braintreebank_movies/'
# movies_list = [os.path.join(MOVIES_DIR, f) for f in os.listdir(MOVIES_DIR) if os.path.isfile(os.path.join(MOVIES_DIR, f))]
movies_list = [os.path.join(MOVIES_DIR, f) for f in os.listdir(MOVIES_DIR) 
               if os.path.isfile(os.path.join(MOVIES_DIR, f)) and f >= 'sesame-street-episode-3990.mp4']

def extract_frame_at_time(video_path, timestamp):
    """
    Extract a frame from video at a specific timestamp.
    
    Args:
        video_path (str): Path to the video file
        timestamp (float): Timestamp in seconds
    Returns:
        PIL.Image: The extracted frame as PIL Image
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Warning: Could not extract frame at timestamp {timestamp}")
        return None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    return pil_image

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

    print("Loading DINOv2 model...")
    device = torch.device('cuda')
    model_name = "facebook/dinov2-base"  # You can change to dinov2-large or dinov2-small
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    output_dir = Path("/om2/data/public/braintreebank_movies_dinov2_preprocessed/")
    output_dir.mkdir(exist_ok=True)

    for movie_path in tqdm(movies_list, desc="Movies"):
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

        for idx in tqdm(frame_indices, desc=f"Frames in {os.path.basename(movie_path)}", leave=False):
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
            movie_name = os.path.splitext(os.path.basename(movie_path))[0]
            np.save(output_dir / f"{movie_name}_dinov2_features.npy", features_array)
            np.save(output_dir / f"{movie_name}_timestamps.npy", timestamps_array)
        else:
            print(f"No frames extracted for {movie_path}") 