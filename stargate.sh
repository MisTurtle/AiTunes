#!/bin/bash

# -- This script is to be run on GPU nodes
# -- It will execute in parallel main.py appended with parameters written in task_params.txt
# -- One process will be created for each line in that file

# Check if a file is provided
if [ "$#" -ne 2 ]; then
    >&2 echo "Usage: $0 <command> <file>"
    exit 1
fi

# Store background process PIDs
pids=()

# Function to clean up child processes on exit
cleanup() {
    echo "Terminating all child processes..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait  # Ensure all processes exit
    exit 1
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Get # of gpu
gpu_count=$(nvidia-smi -L | wc -l)
echo "Found $gpu_count GPUs"

# Read file line by line, ignoring empty lines and lines starting with #
i=0
while IFS='' read -r line; do

    [[ -z "$line" || "$line" =~ ^# ]] && continue  # Skip empty lines and comments

    # Assign a different gpu for each task
    gpu_id=$((i % gpu_count))

    # Run each line as a command in the background
    CUDA_VISIBLE_DEVICES=$gpu_id eval "$1 $line" &
    pid=$!
    pids+=("$pid")  # Store PID

    echo "Spawned:"
    echo " - Command: $1 $line"
    echo " - PID: $pid"
    echo " - GPU: $gpu_id"
    
    ((i++))

done < "$2"

# Wait for all background processes to finish
wait
echo "All processes finished."