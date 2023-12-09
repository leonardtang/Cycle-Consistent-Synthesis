#!/bin/bash
# Go ham with evaluation...

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null
then
    echo "CUDA is not available on this machine."
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
eval "$(conda shell.bash hook)"
conda activate redteam

# Function to check and run the script on a specific GPU
run_on_gpu() {
    local gpu_id=$1
    echo "Trying to run on GPU $gpu_id"
    
    # Set CUDA_VISIBLE_DEVICES to the current GPU
    # Do some logging or something
    CUDA_VISIBLE_DEVICES=$gpu_id python cycle.py && exit 0
}

# Get the number of GPUs
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Outer loop to iterate through HPs

# Iterate through GPUs and try to run the script
for (( gpu_id=1; gpu_id<num_gpus; gpu_id++ ))
do
    run_on_gpu $gpu_id
done