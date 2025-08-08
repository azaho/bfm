#!/bin/bash
#SBATCH --job-name=movie_preprocessing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --output runs/logs/movie_preprocessing_%A_%a.out
#SBATCH --error runs/logs/movie_preprocessing_%A_%a.err
#SBATCH --array=1-100  # Adjust based on number of movies
#SBATCH -p normal

set -e  # Exit on any error

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default values for environment variables
export MOVIES_DIR=${MOVIES_DIR:-"/om2/data/public/braintreebank_movies"}
export OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"/om2/data/public/braintreebank_movies_pipeline"}
export FRAME_INTERVAL=${FRAME_INTERVAL:-"1.0"}
export AUDIO_FRAME_INTERVAL=${AUDIO_FRAME_INTERVAL:-"1.0"}
export AUDIO_SEGMENT_LENGTH=${AUDIO_SEGMENT_LENGTH:-"0.25"}
export AUDIO_MODEL=${AUDIO_MODEL:-"facebook/wav2vec2-base-960h"}

# Validate required environment variables
if [ -z "$MOVIES_DIR" ]; then
    echo "Error: MOVIES_DIR environment variable is not set"
    echo "Please set it in your .env file or export it directly"
    exit 1
fi

if [ ! -d "$MOVIES_DIR" ]; then
    echo "Error: Movies directory does not exist: $MOVIES_DIR"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_BASE_DIR/clip_features"
mkdir -p "$OUTPUT_BASE_DIR/dinov2_features"
mkdir -p "$OUTPUT_BASE_DIR/audio_features"

# Find all movie files (common video formats)
echo "Scanning for movie files in $MOVIES_DIR..."
movie_files=()
while IFS= read -r -d '' file; do
    movie_files+=("$file")
done < <(find "$MOVIES_DIR" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.wmv" -o -iname "*.flv" -o -iname "*.webm" -o -iname "*.m4v" \) -print0)

total_movies=${#movie_files[@]}
echo "Found $total_movies movie files"

# Calculate which movie to process based on SLURM_ARRAY_TASK_ID
movie_idx=$((SLURM_ARRAY_TASK_ID - 1))

if [ $movie_idx -ge $total_movies ]; then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds number of movies ($total_movies), exiting"
    exit 0
fi

movie_path="${movie_files[$movie_idx]}"
movie_name=$(basename "$movie_path" | sed 's/\.[^.]*$//')

echo "Processing movie $((movie_idx + 1))/$total_movies: $movie_name"
echo "Configuration:"
echo "  Movies directory: $MOVIES_DIR"
echo "  Output base directory: $OUTPUT_BASE_DIR"
echo "  Frame interval: $FRAME_INTERVAL"
echo "  Audio frame interval: $AUDIO_FRAME_INTERVAL"
echo "  Audio segment length: $AUDIO_SEGMENT_LENGTH"
echo "  Audio model: $AUDIO_MODEL"
echo ""

# Function to process a single preprocessing pipeline
process_pipeline() {
    local pipeline_name="$1"
    local movie_path="$2"
    local movie_name="$3"
    local frame_interval="$4"
    local output_dir="$5"
    local script_name="$6"
    local additional_args="$7"
    
    local output_path="$output_dir/${movie_name}_${pipeline_name}_features.npz"
    
    if [ ! -f "$output_path" ]; then
        echo "  Extracting $pipeline_name features for $movie_name..."
        python "$script_name" \
            --movie_path "$movie_path" \
            --frame_interval "$frame_interval" \
            --output_path "$output_path" \
            $additional_args
        echo "  Completed $pipeline_name features for $movie_name"
    else
        echo "  $pipeline_name features already exist for $movie_name, skipping..."
    fi
}

# Process all three pipelines in parallel
echo "Starting parallel processing of all pipelines for $movie_name..."

# CLIP features
process_pipeline "clip" "$movie_path" "$movie_name" "$FRAME_INTERVAL" "$OUTPUT_BASE_DIR/clip_features" "clip_preprocessing.py" "" &

# DINOv2 features  
process_pipeline "dinov2" "$movie_path" "$movie_name" "$FRAME_INTERVAL" "$OUTPUT_BASE_DIR/dinov2_features" "dinov2_preprocessing.py" "" &

# Audio features
process_pipeline "audio" "$movie_path" "$movie_name" "$AUDIO_FRAME_INTERVAL" "$OUTPUT_BASE_DIR/audio_features" "audio_preprocessing.py" "--segment_length $AUDIO_SEGMENT_LENGTH --model_name $AUDIO_MODEL" &

# Wait for all pipelines to complete
wait

echo "Completed processing all pipelines for $movie_name"
echo "Output files:"
echo "  CLIP: $OUTPUT_BASE_DIR/clip_features/${movie_name}_clip_features.npz"
echo "  DINOv2: $OUTPUT_BASE_DIR/dinov2_features/${movie_name}_dinov2_features.npz"
echo "  Audio: $OUTPUT_BASE_DIR/audio_features/${movie_name}_audio_features.npz" 