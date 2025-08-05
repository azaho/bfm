import os
import pandas as pd
import numpy as np
import torch
import cv2
from tqdm import tqdm
import argparse
from transformers import AutoFeatureExtractor, AutoModel
import torchaudio
from moviepy import VideoFileClip

def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio features from movie files.")
    parser.add_argument(
        "--segment_interval",
        type=float,
        default=None,
        help="Time interval (in seconds) between consecutive audio segments to extract. "
             "If not specified, uses spectrogram-compatible hop length (0.0625s for 2048Hz)."
    )
    parser.add_argument(
        "--movie_path",
        type=str,
        required=True,
        help="Path to the movie or audio file to process."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where to save the resulting file (should end with .npz)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base",
        help="Audio model to use for feature extraction. Options: facebook/wav2vec2-base, "
             "facebook/wav2vec2-large, openai/whisper-base, etc."
    )
    return parser.parse_args()

def extract_audio_segment(video_path, start_time, duration, sample_rate=16000):
    """
    Extract an audio segment from video or audio file at a specific time.
    
    Args:
        video_path (str): Path to the video or audio file
        start_time (float): Start time in seconds
        duration (float): Duration of segment in seconds
        sample_rate (int): Target sample rate for audio
    
    Returns:
        np.ndarray: Audio segment as numpy array
    """
    try:
        # Check if it's a video file by trying to open with OpenCV
        cap = cv2.VideoCapture(video_path)
        is_video = cap.isOpened()
        cap.release()
        
        if is_video:
            # For video files, use moviepy which is more reliable
            try:
                # Load the video file
                video = VideoFileClip(video_path)
                
                # Extract the audio segment
                audio_segment = video.audio.subclip(start_time, start_time + duration)
                
                # Convert to numpy array and resample
                audio_array = audio_segment.to_soundarray(fps=sample_rate)
                
                # Convert to mono if stereo
                if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                    audio_array = np.mean(audio_array, axis=1)
                
                # Close the video to free memory
                video.close()
                
                return audio_array
                
            except Exception as e:
                print(f"Warning: moviepy failed for video file at time {start_time}: {e}")
                return None
        else:
            # For audio files, use torchaudio directly
            try:
                # Get the original sample rate first
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
                print(f"Warning: torchaudio failed for audio file at time {start_time}: {e}")
                return None
    
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
        # For wav2vec2 models - use the last hidden state
        inputs = feature_extractor(audio_segment, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the last hidden state and take mean across time dimension
            # This is the recommended approach for wav2vec2 feature extraction
            features = outputs.last_hidden_state.mean(dim=1)  # Shape: (1, hidden_size)
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
    
    elif "whisper" in model.config.model_type:
        # For whisper models - use the encoder output
        inputs = feature_extractor(audio_segment, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use the last encoder hidden state and take mean across time dimension
            # This is the recommended approach for Whisper feature extraction
            features = outputs.encoder_last_hidden_state.mean(dim=1)  # Shape: (1, hidden_size)
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
    
    elif "hubert" in model.config.model_type:
        # For HuBERT models - use the last hidden state
        inputs = feature_extractor(audio_segment, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the last hidden state and take mean across time dimension
            features = outputs.last_hidden_state.mean(dim=1)  # Shape: (1, hidden_size)
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
            elif hasattr(outputs, 'encoder_last_hidden_state'):
                features = outputs.encoder_last_hidden_state.mean(dim=1)
            else:
                # Fallback: use the first output tensor
                features = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.mean(dim=1)
            
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu()

def get_video_duration(video_path):
    """
    Get the duration of a video or audio file.
    
    Args:
        video_path (str): Path to the video or audio file
    
    Returns:
        float: Duration in seconds
    """
    try:
        # Check if it's a video file by trying to open with OpenCV
        cap = cv2.VideoCapture(video_path)
        is_video = cap.isOpened()
        cap.release()
        
        if is_video:
            # For video files, use moviepy
            try:
                video = VideoFileClip(video_path)
                duration = video.duration
                video.close()
                print(f"Video duration: {duration:.2f}s (from moviepy)")
                return duration
            except Exception as e:
                print(f"moviepy failed: {e}")
                # Fallback to OpenCV
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                if fps > 0 and total_frames > 0:
                    duration = total_frames / fps
                    print(f"Video duration: {duration:.2f}s (from OpenCV: {total_frames} frames at {fps:.2f} fps)")
                    return duration
        
        # If OpenCV fails, try torchaudio for audio files
        try:
            metadata = torchaudio.info(video_path)
            duration = metadata.num_frames / metadata.sample_rate
            print(f"Audio duration: {duration:.2f}s (from {metadata.num_frames} frames at {metadata.sample_rate} Hz)")
            return duration
        except Exception as audio_error:
            print(f"torchaudio failed: {audio_error}")
            
        raise Exception("Could not determine duration with either moviepy, OpenCV, or torchaudio")
        
    except Exception as e:
        print(f"Warning: Could not get duration for {video_path}: {e}")
        return None

def calculate_spectrogram_compatible_interval(sampling_rate=2048):
    """
    Calculate the segment interval that matches the spectrogram hop length.
    
    Args:
        sampling_rate (int): Audio sampling rate
    
    Returns:
        float: Segment interval in seconds
    """
    # Spectrogram parameters from the config
    tperseg = 0.25  # seconds per segment
    poverlap = 0.75  # 75% overlap
    
    # Calculate hop length
    nperseg = int(tperseg * sampling_rate)
    noverlap = int(poverlap * nperseg)
    hop_length = nperseg - noverlap
    
    # Convert to seconds
    hop_interval = hop_length / sampling_rate
    
    return hop_interval

if __name__ == "__main__":
    args = parse_args()
    segment_interval = args.segment_interval
    movie_path = args.movie_path
    output_path = args.output_path
    model_name = args.model_name

    # Validate input file exists
    if not os.path.exists(movie_path):
        raise FileNotFoundError(f"Movie/audio file not found: {movie_path}")

    # Validate output path ends with .npz
    if not output_path.endswith('.npz'):
        output_path = output_path + '.npz'

    print(f"Loading audio model: {model_name}")
    device = torch.device('cuda')
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Determine segment_interval if not provided
    if segment_interval is None:
        # Calculate the spectrogram-compatible hop length
        # This ensures the number of audio features matches the number of spectrogram time bins
        segment_interval = calculate_spectrogram_compatible_interval(sampling_rate=2048)
        print(f"Using spectrogram-compatible segment_interval: {segment_interval}s (for 2048Hz sampling rate)")

    print(f"Processing movie/audio: {movie_path}")
    duration = get_video_duration(movie_path)
    if duration is None:
        print(f"Skipping {movie_path} - could not get duration")
        exit()

    features_list = []
    timestamps = []

    # Extract audio segments at specified time intervals
    segment_times = np.arange(0, duration, segment_interval)
    
    print(f"Extracting {len(segment_times)} audio segments with interval {segment_interval}s")
    
    for start_time in tqdm(segment_times, desc=f"Audio segments in {os.path.basename(movie_path)}"):
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
        
        print(f"Features shape: {features_array.shape}")
        print(f"Timestamps shape: {timestamps_array.shape}")
        
        # Save both features and timestamps in a single npz file
        np.savez(output_path, 
                 features=features_array, 
                 timestamps=timestamps_array)
        print(f"Saved features and timestamps to {output_path}")
    else:
        print(f"No audio segments extracted for {movie_path}")


##### TO RUN ######
# For spectrogram-compatible features (recommended):
# python audio_preprocessing.py --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_wav2vec2_features.npz --model_name facebook/wav2vec2-base

# For custom interval:
# python audio_preprocessing.py --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --segment_interval 0.0625 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_wav2vec2_features.npz --model_name facebook/wav2vec2-base

# For Whisper model:
# python audio_preprocessing.py --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_whisper_features.npz --model_name openai/whisper-base
##################### 