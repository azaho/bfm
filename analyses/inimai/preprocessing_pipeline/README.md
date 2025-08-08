# Movie Feature Extraction Pipeline

This directory contains scripts to extract features from movie files using three different models:
- **CLIP**: Visual features using OpenAI's CLIP model
- **DINOv2**: Visual features using Meta's DINOv2 model  
- **Audio**: Audio features using wav2vec2 or other audio models

## Scripts

### Individual Processing Scripts

1. **`clip_preprocessing.py`** - Extract CLIP features from movie frames
2. **`dinov2_preprocessing.py`** - Extract DINOv2 features from movie frames
3. **`audio_preprocessing.py`** - Extract audio features from movie audio

### Batch Processing Scripts

**`submit_movie_processing.sh`** - Submit SLURM job to process all movies in parallel using all three pipelines
**`process_all_movies.sh`** - SLURM job script (used internally by submit script)

## Quick Start

### 1. Set up environment variables

Create a `.env` file in this directory with your configuration:

```bash
# Copy the example configuration
cp env_example.txt .env

# Edit the .env file with your paths
nano .env
```

Example `.env` file:
```bash
MOVIES_DIR=/om2/data/public/braintreebank_movies
OUTPUT_BASE_DIR=/om2/data/public/braintreebank_movies_preprocessed
FRAME_INTERVAL=0.25
AUDIO_FRAME_INTERVAL=0.0625
AUDIO_SEGMENT_LENGTH=0.25
AUDIO_MODEL=facebook/wav2vec2-base-960h
MAX_PARALLEL_JOBS=4
```

### 2. Make the script executable

```bash
chmod +x submit_movie_processing.sh
```

### 3. Submit the batch processing job

```bash
./submit_movie_processing.sh
```

## Configuration Options

### Environment Variables

- **`MOVIES_DIR`**: Directory containing movie files (required)
- **`OUTPUT_BASE_DIR`**: Base directory for output files (default: `/om2/data/public/braintreebank_movies_preprocessed`)
- **`FRAME_INTERVAL`**: Time interval between frames for visual features in seconds (default: 0.25)
- **`AUDIO_FRAME_INTERVAL`**: Time interval for audio segments in seconds (default: 0.0625)
- **`AUDIO_SEGMENT_LENGTH`**: Length of audio segments in seconds (default: 0.25)
- **`AUDIO_MODEL`**: Audio model to use (default: `facebook/wav2vec2-base-960h`)
- **`MAX_PARALLEL_JOBS`**: Maximum number of parallel jobs (default: 4) - Note: Not used in SLURM version

### Supported Audio Models

- `facebook/wav2vec2-base-960h` (default)
- `facebook/wav2vec2-large-960h`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `facebook/hubert-base-ls960`

### Supported Video Formats

- MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V

## Output Structure

The script creates the following directory structure:

```
OUTPUT_BASE_DIR/
├── clip_features/
│   ├── movie1_clip_features.npz
│   ├── movie2_clip_features.npz
│   └── ...
├── dinov2_features/
│   ├── movie1_dinov2_features.npz
│   ├── movie2_dinov2_features.npz
│   └── ...
└── audio_features/
    ├── movie1_audio_features.npz
    ├── movie2_audio_features.npz
    └── ...
```

Each `.npz` file contains:
- `features`: Extracted features as numpy array
- `timestamps`: Corresponding timestamps as numpy array

## Individual Script Usage

### CLIP Features
```bash
python clip_preprocessing.py \
    --movie_path /path/to/movie.mp4 \
    --frame_interval 0.25 \
    --output_path /path/to/output.npz
```

### DINOv2 Features
```bash
python dinov2_preprocessing.py \
    --movie_path /path/to/movie.mp4 \
    --frame_interval 0.25 \
    --output_path /path/to/output.npz
```

### Audio Features
```bash
python audio_preprocessing.py \
    --frame_interval 0.0625 \
    --segment_length 0.25 \
    --movie_path /path/to/movie.mp4 \
    --output_path /path/to/output.npz \
    --model_name facebook/wav2vec2-base-960h
```

## Features

- **SLURM Job Array**: Uses SLURM job arrays for maximum parallelization
- **Triple Parallel Processing**: Each movie has all three pipelines (CLIP, DINOv2, Audio) processed simultaneously
- **Resume Capability**: Skips already processed files
- **Dataset Agnostic**: Works with any movie dataset by changing environment variables
- **Error Handling**: Continues processing even if individual movies fail
- **Progress Tracking**: Shows progress for each movie and overall completion
- **Automatic Array Sizing**: Automatically determines the correct SLURM array size based on number of movies

## Requirements

- Python 3.7+
- PyTorch
- torchaudio
- transformers
- opencv-python
- moviepy
- PIL (Pillow)
- clip
- numpy
- tqdm

## SLURM Configuration

The script uses the following SLURM resources per movie:
- **GPU**: 1x A100 GPU
- **CPU**: 8 cores
- **Memory**: 64GB
- **Time Limit**: 24 hours
- **Partition**: normal

## Notes

- The script automatically detects video vs audio files and uses appropriate processing methods
- Audio features use overlapping segments for better temporal resolution
- Visual features are extracted at regular time intervals
- All features are L2-normalized for consistency
- Each SLURM job processes one movie with all three pipelines running in parallel
- The submit script automatically counts movies and sets the correct array size 