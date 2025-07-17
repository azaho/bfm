import os
import json
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
import clip
from tqdm import tqdm
import argparse
from pathlib import Path

def extract_frame_at_time(video_path, timestamp, output_path=None):
    """
    Extract a frame from video at a specific timestamp.
    
    Args:
        video_path (str): Path to the video file
        timestamp (float): Timestamp in seconds
        output_path (str, optional): Path to save the frame image
    
    Returns:
        PIL.Image: The extracted frame as PIL Image
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    
    # Set position to the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Warning: Could not extract frame at timestamp {timestamp}")
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    if output_path:
        pil_image.save(output_path)
    
    return pil_image

def get_clip_features(model, preprocess, image, device):
    """
    Extract CLIP features from an image.
    
    Args:
        model: CLIP model
        preprocess: CLIP preprocessing function
        image: PIL Image
        device: torch device
    
    Returns:
        torch.Tensor: CLIP features
    """
    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu()

def main():
    parser = argparse.ArgumentParser(description='Extract CLIP features from movie screenshots')
    parser.add_argument('--subject_id', type=int, default=10, help='Subject ID')
    parser.add_argument('--trial_id', type=int, default=1, help='Trial ID')
    parser.add_argument('--movie_path', type=str, default='spider-man-far-from-home.mp4', help='Path to movie file')
    parser.add_argument('--output_dir', type=str, default='clip_features', help='Output directory for features')
    parser.add_argument('--save_frames', action='store_true', help='Save extracted frames as images')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.save_frames:
        frames_dir = output_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
    
    # Load CLIP model
    print("Loading CLIP model...")
    device = torch.device(args.device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load timing data
    csv_path = f"evaluation/neuroprobe/braintreebank_features_time_alignment/subject{args.subject_id}_trial{args.trial_id}_words_df.csv"
    print(f"Loading timing data from {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Extract features for each timestamp
    features_list = []
    timestamps = []
    
    print(f"Extracting CLIP features for {len(df)} timestamps...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        timestamp = row['start']  # Use start time of each word
        
        # Extract frame
        frame_path = None
        if args.save_frames:
            frame_path = frames_dir / f"frame_{idx:06d}_{timestamp:.3f}.jpg"
        
        frame = extract_frame_at_time(args.movie_path, timestamp, frame_path)
        
        if frame is not None:
            # Extract CLIP features
            features = get_clip_features(model, preprocess, frame, device)
            
            features_list.append(features.squeeze().numpy())
            timestamps.append(timestamp)
        else:
            print(f"Warning: Could not extract frame for timestamp {timestamp}")
    
    # Convert to numpy array
    features_array = np.array(features_list)
    
    # Save features
    features_path = output_dir / f"clip_features_subject{args.subject_id}_trial{args.trial_id}.npy"
    timestamps_path = output_dir / f"timestamps_subject{args.subject_id}_trial{args.trial_id}.npy"
    
    np.save(features_path, features_array)
    np.save(timestamps_path, np.array(timestamps))
    
    # Save metadata
    metadata = {
        'subject_id': args.subject_id,
        'trial_id': args.trial_id,
        'movie_path': args.movie_path,
        'num_frames': len(features_list),
        'feature_dim': features_array.shape[1],
        'model': 'ViT-B/32',
        'timestamps': timestamps
    }
    
    metadata_path = output_dir / f"metadata_subject{args.subject_id}_trial{args.trial_id}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(features_list)} CLIP features to {features_path}")
    print(f"Feature shape: {features_array.shape}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main() 