import os
import json
import pandas as pd
import numpy as np
import torch
import librosa
import torchaudio
import subprocess
import tempfile
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio features from movie or audio files.")
    parser.add_argument(
        "--frame_interval",
        type=float,
        required=True,
        help="Time interval (in seconds) between consecutive audio segment starts."
    )
    parser.add_argument(
        "--hop_length",
        type=float,
        default=None,
        help="Length of each audio segment (in seconds) to extract features from. "
             "If not specified, defaults to frame_interval (no overlap)."
    )
    parser.add_argument(
        "--movie_path",
        type=str,
        required=True,
        help="Path to the movie/audio file to process."
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
        default="facebook/wav2vec2-base-960h",
        help="Pretrained audio model to use for feature extraction."
    )
    return parser.parse_args()

def get_audio_features(model, processor, audio_segment, device, sample_rate=16000):
    """
    Extract audio features from an audio segment using Wav2Vec2.
    
    Args:
        model: Wav2Vec2 model
        processor: Wav2Vec2 processor
        audio_segment: numpy array of audio samples
        device: torch device
        sample_rate: sample rate of audio
    
    Returns:
        torch.Tensor: Audio features from the last hidden state
    """
    # Ensure audio is the right length and format
    if len(audio_segment) == 0:
        return None
    
    # Process audio - wav2vec2 expects 16kHz
    inputs = processor(audio_segment, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs['input_values'].to(device)
    
    with torch.no_grad():
        # Get the last hidden state (this is what's typically used for downstream tasks)
        outputs = model(input_values)
        # Take mean across time dimension to get a fixed-size representation
        features = outputs.last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)
    
    return features.cpu()

def load_audio(file_path, target_sr=16000):
    """
    Load audio from video or audio file using FFmpeg as primary method.
    
    Args:
        file_path: path to audio/video file
        target_sr: target sample rate
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    # First try FFmpeg for video files (most reliable for MP4, etc.)
    try:
        # Create temporary file for extracted audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Use FFmpeg to extract audio and convert to WAV
        cmd = [
            'ffmpeg', '-i', file_path,
            '-ar', str(target_sr),  # Set sample rate
            '-ac', '1',  # Convert to mono
            '-f', 'wav',  # Output format
            '-y',  # Overwrite output file
            temp_path
        ]
        
        # Run FFmpeg with suppressed output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Load the extracted audio with librosa
            audio_data, sr = librosa.load(temp_path, sr=target_sr, mono=True)
            os.unlink(temp_path)  # Clean up temp file
            return audio_data, sr
        else:
            os.unlink(temp_path)  # Clean up temp file
            raise Exception(f"FFmpeg failed: {result.stderr}")
            
    except Exception as e:
        print(f"FFmpeg extraction failed: {e}")
        
        # Fallback to librosa for audio files
        try:
            audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio_data, sr
        except Exception as e2:
            print(f"Error loading with librosa: {e2}")
            
            # Final fallback to torchaudio
            try:
                audio_data, sr = torchaudio.load(file_path)
                # Convert to mono if stereo
                if audio_data.shape[0] > 1:
                    audio_data = torch.mean(audio_data, dim=0)
                else:
                    audio_data = audio_data.squeeze(0)
                
                # Resample if needed
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    audio_data = resampler(audio_data)
                
                return audio_data.numpy(), target_sr
            except Exception as e3:
                raise Exception(f"Could not load audio with FFmpeg, librosa, or torchaudio: {e}, {e2}, {e3}")

if __name__ == "__main__":
    args = parse_args()
    frame_interval = args.frame_interval  # time between segment starts
    segment_length = args.hop_length if args.hop_length is not None else frame_interval  # length of each segment
    movie_path = args.movie_path
    output_path = args.output_path
    model_name = args.model_name

    # Validate input file exists
    if not os.path.exists(movie_path):
        raise FileNotFoundError(f"Audio/Movie file not found: {movie_path}")

    # Validate output path ends with .npz
    if not output_path.endswith('.npz'):
        output_path = output_path + '.npz'

    print(f"Loading audio model: {model_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    print(f"Loading audio from: {movie_path}")
    audio_data, sample_rate = load_audio(movie_path, target_sr=16000)
    duration = len(audio_data) / sample_rate
    
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")

    # Configure segmentation based on arguments
    hop_samples = int(frame_interval * sample_rate)  # samples between segment starts
    segment_samples = int(segment_length * sample_rate)  # samples per segment
    
    overlap_ratio = 1 - (frame_interval / segment_length) if segment_length > 0 else 0
    
    features_list = []
    timestamps = []

    print(f"Using segmentation: {segment_length}s segments with {frame_interval}s interval ({overlap_ratio*100:.1f}% overlap)")
    
    # Generate start positions for each segment
    start_positions = np.arange(0, len(audio_data) - segment_samples + 1, hop_samples)
    
    for start_pos in tqdm(start_positions, desc=f"Processing audio segments from {os.path.basename(movie_path)}"):
        end_pos = start_pos + segment_samples
        audio_segment = audio_data[start_pos:end_pos]
        
        if len(audio_segment) < segment_samples:
            # Pad if necessary
            audio_segment = np.pad(audio_segment, (0, segment_samples - len(audio_segment)), mode='constant')
        
        features = get_audio_features(model, processor, audio_segment, device, sample_rate)
        if features is not None:
            features_list.append(features.squeeze(0).numpy())
            # Timestamp is the center of the segment
            center_time = (start_pos + segment_samples // 2) / sample_rate
            timestamps.append(center_time)

    print(f"Extracted {len(features_list)} audio segments from {movie_path}")

    if features_list:
        features_array = np.stack(features_list)
        timestamps_array = np.array(timestamps)

        print(f"Features shape: {features_array.shape}")
        print(f"Timestamps shape: {timestamps_array.shape}")
        
        # Save both features and timestamps in a single npz file
        # Using dictionary-style saving as requested
        save_dict = {
            'features': features_array,
            'timestamps': timestamps_array,
            'sample_rate': sample_rate,
            'segment_length': segment_length,
            'frame_interval': frame_interval,
            'overlap_ratio': overlap_ratio,
            'model_name': model_name
        }
        
        np.savez(output_path, **save_dict)
        print(f"Saved features and timestamps to {output_path}")
        
        # Print some statistics
        print(f"Feature dimension: {features_array.shape[1]}")
        print(f"Time coverage: {timestamps_array[0]:.2f}s to {timestamps_array[-1]:.2f}s")
        if len(timestamps_array) > 1:
            avg_interval = np.mean(np.diff(timestamps_array))
            print(f"Average interval between segments: {avg_interval:.3f}s")
    else:
        print(f"No audio segments extracted from {movie_path}")


##### TO RUN ######
# NOTE: Requires FFmpeg to be installed for video file support:
# conda install ffmpeg -c conda-forge
# or: sudo apt-get install ffmpeg (on Ubuntu/Debian)
#
# Example 1: 0.25s intervals with 0.25s segments (no overlap, hop_length defaults to frame_interval)
# python audio_preprocessing.py --frame_interval 0.25 --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_audio_features.npz
# 
# Example 2: 0.0625s intervals with 0.25s segments (75% overlap, like spectrogram parameters)
# python audio_preprocessing.py --frame_interval 0.0625 --hop_length 0.25 --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_audio_features.npz
#
# Example 3: 0.5s intervals with 1.0s segments (50% overlap)
# python audio_preprocessing.py --frame_interval 0.5 --hop_length 1.0 --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_audio_features.npz
#####################