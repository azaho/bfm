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
import librosa
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel
import torchaudio

def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio features from movie files.")
    parser.add_argument(
        "--segment_interval",
        type=float,
        default=1.0,
        help="Time interval (in seconds) between consecutive audio segments to extract. "
             "Default is 1.0 second."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base",
        help="Audio model to use for feature extraction. Options: facebook/wav2vec2-base, "
             "facebook/wav2vec2-large, openai/whisper-base, etc."
    )
    return parser.parse_args()

MOVIES_DIR = '/om2/data/public/braintreebank_movies/'
# movies_list = [os.path.join(MOVIES_DIR, f) for f in os.listdir(MOVIES_DIR) if os.path.isfile(os.path.join(MOVIES_DIR, f))]
movies_list = [os.path.join(MOVIES_DIR, f) for f in os.listdir(MOVIES_DIR) 
               if os.path.isfile(os.path.join(MOVIES_DIR, f)) and f >= 'sesame-street-episode-3990.mp4']

def extract_audio_segment(video_path, start_time, duration, sample_rate=16000):
    """
    Extract an audio segment from video at a specific time.
    
    Args:
        video_path (str): Path to the video file
        start_time (float): Start time in seconds
        duration (float): Duration of segment in seconds
        sample_rate (int): Target sample rate for audio
    
    Returns:
        np.ndarray: Audio segment as numpy array
    """
    try:
        # First get the audio info to determine the original sample rate
        metadata = torchaudio.info(video_path)
        original_sr = metadata.sample_rate
        
        # Calculate frame positions
        start_frame = int(start_time * original_sr)
        num_frames = int(duration * original_sr)
        
        # Load the audio segment
        waveform, sr = torchaudio.load(video_path, frame_offset=start_frame, num_frames=num_frames)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze().numpy()
    
    except Exception as e:
        print(f"Warning: Could not extract audio segment at time {start_time}: {e}")
        return None

def get_audio_features(model, feature_extractor, audio_segment, device):
    """
    Extract audio features from an audio segment.
    
    Args:
        model: Audio model (e.g., wav2vec2, whisper)
        feature_extractor: Audio feature extractor
        audio_segment: numpy array of audio
        device: torch device
    
    Returns:
        torch.Tensor: Audio features
    """
    # Prepare inputs based on model type
    if "wav2vec2" in model.config.model_type:
        # For wav2vec2 models
        inputs = feature_extractor(audio_segment, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the last hidden state and take mean across time dimension
            features = outputs.last_hidden_state.mean(dim=1)  # Shape: (1, hidden_size)
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
    
    elif "whisper" in model.config.model_type:
        # For whisper models
        inputs = feature_extractor(audio_segment, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use the last encoder hidden state and take mean across time dimension
            features = outputs.encoder_last_hidden_state.mean(dim=1)  # Shape: (1, hidden_size)
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
    
    else:
        # Generic approach for other models
        inputs = feature_extractor(audio_segment, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Try to get features from common output attributes
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'hidden_states'):
                features = outputs.hidden_states[-1].mean(dim=1)
            else:
                # Fallback: use the first output tensor
                features = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.mean(dim=1)
            
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu()

def get_video_duration(video_path):
    """
    Get the duration of a video file.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        float: Duration in seconds
    """
    try:
        # Use torchaudio to get duration
        metadata = torchaudio.info(video_path)
        return metadata.num_frames / metadata.sample_rate
    except Exception as e:
        print(f"Warning: Could not get duration for {video_path}: {e}")
        return None

if __name__ == "__main__":
    args = parse_args()
    segment_interval = args.segment_interval
    model_name = args.model_name

    print(f"Loading audio model: {model_name}")
    device = torch.device('cuda')
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    output_dir = Path(f"/om2/data/public/braintreebank_movies_audio_preprocessed_{model_name.split('/')[-1]}/")
    output_dir.mkdir(exist_ok=True)

    for movie_path in tqdm(movies_list, desc="Movies"):
        duration = get_video_duration(movie_path)
        if duration is None:
            print(f"Skipping {movie_path} - could not get duration")
            continue

        features_list = []
        timestamps = []

        # Extract audio segments at specified time intervals
        segment_times = np.arange(0, duration, segment_interval)
        
        for start_time in tqdm(segment_times, desc=f"Audio segments in {os.path.basename(movie_path)}", leave=False):
            # Extract audio segment
            audio_segment = extract_audio_segment(movie_path, start_time, segment_interval)
            
            if audio_segment is None or len(audio_segment) == 0:
                continue
            
            # Get features
            features = get_audio_features(model, feature_extractor, audio_segment, device)
            features_list.append(features.squeeze(0).numpy())
            timestamps.append(start_time)

        print(f"Extracted audio segments for {movie_path}")

        if features_list:
            features_array = np.stack(features_list)
            timestamps_array = np.array(timestamps)
            movie_name = os.path.splitext(os.path.basename(movie_path))[0]
            model_suffix = model_name.split('/')[-1]
            np.save(output_dir / f"{movie_name}_{model_suffix}_features.npy", features_array)
            np.save(output_dir / f"{movie_name}_timestamps.npy", timestamps_array)
        else:
            print(f"No audio segments extracted for {movie_path}") 