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
    parser = argparse.ArgumentParser(description="Extract audio features from movie or audio files.")
    parser.add_argument(
        "--frame_interval",
        type=float,
        default=1.0,
        help="Time interval (in seconds) between consecutive audio segment starts."
    )
    parser.add_argument(
        "--segment_length",
        type=float,
        default=0.25,
        help="Length of each audio segment (in seconds) to extract features from. "
             "Defaults to 0.25 seconds (250ms)."
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

def get_model_sample_rate(feature_extractor):
    """
    Get the expected sampling rate for the model from the feature extractor.
    """
    # Most HuggingFace feature extractors have a .sampling_rate attribute
    if hasattr(feature_extractor, "sampling_rate"):
        return feature_extractor.sampling_rate
    # Some have .config.sampling_rate
    if hasattr(feature_extractor, "config") and hasattr(feature_extractor.config, "sampling_rate"):
        return feature_extractor.config.sampling_rate
    # Fallback to 16000 (common for wav2vec2/whisper)
    return 16000

def extract_audio_segment(video_path, start_time, duration, sample_rate):
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
                video = VideoFileClip(video_path)
                
                # Get the subclip and then its audio
                subclip = video.subclipped(start_time, start_time + duration)
                audio_segment = subclip.audio
                
                # Convert to numpy array and resample
                audio_array = audio_segment.to_soundarray(fps=sample_rate)
                if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                    audio_array = np.mean(audio_array, axis=1)
                video.close()
                return audio_array
            except Exception as e:
                print(f"Warning: moviepy failed for video file at time {start_time}: {e}")
                return None
        else:
            # For audio files, use torchaudio directly
            try:
                metadata = torchaudio.info(video_path)
                original_sr = metadata.sample_rate
                start_frame = int(start_time * original_sr)
                num_frames = int(duration * original_sr)
                waveform, sr = torchaudio.load(video_path, frame_offset=start_frame, num_frames=num_frames)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
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

def get_audio_features(model, feature_extractor, audio_segment, device, model_sample_rate):
    """
    Extract audio features from an audio segment.

    Args:
        model: Audio model (e.g., wav2vec2, whisper)
        feature_extractor: Audio feature extractor
        audio_segment: numpy array of audio
        device: torch device
        model_sample_rate: int, expected sample rate for the model

    Returns:
        torch.Tensor: Audio features
    """
    # Prepare inputs based on model type
    if "wav2vec2" in model.config.model_type:
        inputs = feature_extractor(audio_segment, sampling_rate=model_sample_rate, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
    elif "whisper" in model.config.model_type:
        inputs = feature_extractor(audio_segment, sampling_rate=model_sample_rate, return_tensors="pt").to(device)
        # For Whisper, must provide decoder_input_ids or decoder_inputs_embeds
        # We'll use the default decoder_start_token_id for a single token as input
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
        with torch.no_grad():
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
            features = outputs.encoder_last_hidden_state.mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
    elif "hubert" in model.config.model_type:
        inputs = feature_extractor(audio_segment, sampling_rate=model_sample_rate, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
    else:
        inputs = feature_extractor(audio_segment, sampling_rate=model_sample_rate, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'hidden_states'):
                features = outputs.hidden_states[-1].mean(dim=1)
            elif hasattr(outputs, 'encoder_last_hidden_state'):
                features = outputs.encoder_last_hidden_state.mean(dim=1)
            else:
                features = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.mean(dim=1)
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
        cap = cv2.VideoCapture(video_path)
        is_video = cap.isOpened()
        cap.release()

        if is_video:
            try:
                video = VideoFileClip(video_path)
                duration = video.duration
                video.close()
                print(f"Video duration: {duration:.2f}s (from moviepy)")
                return duration
            except Exception as e:
                print(f"moviepy failed: {e}")
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if fps > 0 and total_frames > 0:
                    duration = total_frames / fps
                    print(f"Video duration: {duration:.2f}s (from OpenCV: {total_frames} frames at {fps:.2f} fps)")
                    return duration

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
    tperseg = 0.25
    poverlap = 0.75
    nperseg = int(tperseg * sampling_rate)
    noverlap = int(poverlap * nperseg)
    hop_length = nperseg - noverlap
    hop_interval = hop_length / sampling_rate
    return hop_interval

if __name__ == "__main__":
    args = parse_args()
    frame_interval = args.frame_interval
    segment_length = args.segment_length
    movie_path = args.movie_path
    output_path = args.output_path
    model_name = args.model_name

    if not os.path.exists(movie_path):
        raise FileNotFoundError(f"Movie/audio file not found: {movie_path}")

    if not output_path.endswith('.npz'):
        output_path = output_path + '.npz'

    print(f"Loading audio model: {model_name}")
    device = torch.device('cuda')
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model_sample_rate = get_model_sample_rate(feature_extractor)
    print(f"Model expects sampling rate: {model_sample_rate}")

    print(f"Processing movie/audio: {movie_path}")
    duration = get_video_duration(movie_path)
    if duration is None:
        print(f"Skipping {movie_path} - could not get duration")
        exit()

    features_list = []
    timestamps = []

    overlap_ratio = 1 - (frame_interval / segment_length) if segment_length > 0 else 0

    segment_times = np.arange(0, duration, frame_interval)

    print(f"Extracting {len(segment_times)} audio segments with {segment_length}s segments and {frame_interval}s interval ({overlap_ratio*100:.1f}% overlap)")

    for start_time in tqdm(segment_times, desc=f"Audio segments in {os.path.basename(movie_path)}"):
        audio_segment = extract_audio_segment(movie_path, start_time, segment_length, model_sample_rate)
        if audio_segment is None or len(audio_segment) == 0:
            continue
        features = get_audio_features(model, feature_extractor, audio_segment, device, model_sample_rate)
        features_list.append(features.squeeze(0).numpy())
        timestamps.append(start_time)

    print(f"Extracted audio segments for {movie_path}")

    if features_list:
        features_array = np.stack(features_list)
        timestamps_array = np.array(timestamps)

        print(f"Features shape: {features_array.shape}")
        print(f"Timestamps shape: {timestamps_array.shape}")

        np.savez(output_path,
                 features=features_array,
                 timestamps=timestamps_array)
        print(f"Saved features and timestamps to {output_path}")
    else:
        print(f"No audio segments extracted for {movie_path}")


##### TO RUN ######
# Example 1: 0.25s intervals with 0.25s segments (no overlap, hop_length defaults to frame_interval)
# python audio_preprocessing.py --frame_interval 0.25 --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_audio_features.npz
# 
# Example 2: 0.0625s intervals with 0.25s segments (75% overlap, like spectrogram parameters)
# python audio_preprocessing.py --frame_interval 0.0625 --segment_length 0.25 --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_audio_features.npz
#
# Example 3: 0.5s intervals with 1.0s segments (50% overlap)
# python audio_preprocessing.py --frame_interval 0.5 --segment_length 1.0 --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_audio_features.npz
#
# Example 4: For Whisper model:
# python audio_preprocessing.py --frame_interval 0.25 --movie_path /om2/data/public/braintreebank_movies/ant-man.mp4 --output_path /om2/data/public/braintreebank_movies_audio_preprocessed/ant-man_whisper_features.npz --model_name openai/whisper-base
##################### 
