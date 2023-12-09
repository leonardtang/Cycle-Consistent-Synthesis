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
    local params=$2

    echo "Running on GPU $gpu_id with params: $params"
    
    # Set CUDA_VISIBLE_DEVICES to the current GPU
    # Do some logging or something
    CUDA_VISIBLE_DEVICES=$gpu_id python cycle.py $params
}

# Get the number of GPUs
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Outer loop to iterate through HPs

# Generate hyperparameter combinations and iterate over them
python generate_hps.py | while IFS=, read -r selectcrit fewshot gentemp repeat
do
    for (( gpu_id=0; gpu_id<num_gpus; gpu_id++ ))
    do
        run_on_gpu $gpu_id "--select-crit $selectcrit --few-shot $fewshot --gen-temp $gentemp --repeat $repeat"
    done
done