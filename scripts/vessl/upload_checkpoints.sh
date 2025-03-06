#!/bin/bash

# Parse arguments

CKPT_PATH="checkpoints/llava-v1.5-7b-pperm3x3-lora-r32"
BUCKET_NAME="volume://vessl-storage/sparse-drawer"
BUCKET_PATH="artifacts"


# find "$CKPT_PATH" -type f | while read -r file; do
#   if [[ "$file" != *"/global_step"* ]]; then
#     echo "Copying: $file"
#     vessl storage copy-file "$file" "$BUCKET_NAME/$BUCKET_PATH"
#   fi
# done

# echo "File transfer completed."
# Find all valid files

# Define cloud storage bucket
# BUCKET_NAME="volume://vessl-storage/my-volume"

# Extract the last part of CKPT_PATH
CKPT_NAME=$(basename "$CKPT_PATH")

# Find all valid files
FILES=( $(find "$CKPT_PATH" -type f | grep -v '/global_step') )
TOTAL_FILES=${#FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
  echo "No files to copy."
  exit 0
fi

# Function to display progress
progress_bar() {
  local progress=$(( ($1 * 100) / $2 ))
  local done=$((progress / 2))
  local left=$((50 - done))
  printf "\r["; printf "#%.0s" $(seq 1 $done); printf "%.0s" $(seq 1 $left); printf "] %d%% (%d/%d)" "$progress" "$1" "$2"
}

# Copy files with progress bar
count=0
start_time=$(date +%s)
for file in "${FILES[@]}"; do
  RELATIVE_PATH=${file#"$CKPT_PATH/"}
  CLOUD_PATH="$BUCKET_NAME/$BUCKET_PATH/$CKPT_NAME/$RELATIVE_PATH"
  echo $file
  echo $CLOUD_PATH
  vessl storage copy-file "$file" "$CLOUD_PATH"
  count=$((count + 1))
  elapsed_time=$(( $(date +%s) - start_time ))
  avg_time_per_file=$(( elapsed_time / count ))
  remaining_time=$(( avg_time_per_file * (TOTAL_FILES - count) ))
  progress_bar "$count" "$TOTAL_FILES"
  printf " ETA: %ds" "$remaining_time"
done

echo -e "\nFile transfer completed."