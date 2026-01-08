#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate /aloy/home/acomajuncosa/anaconda3/envs/gardp-kpn

# Get alpha from first argument
alpha=$1

# Determine the directory where run.sh is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Get chunk name
CHUNK_FILE="$SCRIPT_DIR/../data/chunks/chunks.csv"
mapfile -t chunks < "$CHUNK_FILE"
chunk_name="${chunks[$alpha]}"

# Call main.py using an absolute path
python "$SCRIPT_DIR/run.py" --chunk_name "$chunk_name" --output_dir "$SCRIPT_DIR/../results/"