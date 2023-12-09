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

# Function to check if there is enough free memory on the GPU
# This is a simplistic check. Replace 2000 with the approximate memory requirement of your task in MiB.
check_gpu_memory() {
    local gpu_id=$1
    local required_memory=40000
    local free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id)

    if [ "$free_memory" -ge "$required_memory" ]; then
        return 0 # 0 means true/success in Bash
    else
        return 1 # non-zero means false/failure in Bash
    fi
}

# Function to check and run the script on a specific GPU
run_on_gpu() {
    local gpu_id=$1
    local params=$2

    echo "Running on GPU $gpu_id with params: $params"

    # Set CUDA_VISIBLE_DEVICES to the current GPU
    # Replace spaces with underscores and commas with dashes for filename
    local filename=$(echo "$params" | sed 's/ /_/g' | sed 's/,/-/g')

    # TODO: figure out safe way of doing parallel
    CUDA_VISIBLE_DEVICES=$gpu_id python cycle.py $params > "results_${filename}.txt" &
    # CUDA_VISIBLE_DEVICES=$gpu_id python cycle.py $params > "results_${filename}.txt"
}

# Get the number of GPUs
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Generate hyperparameter combinations and iterate over them
python generate_hps.py | while IFS=, read -r selectcrit fewshot gentemp repeat
do
    for (( gpu_id=0; gpu_id<num_gpus; gpu_id++ ))
    do
        if check_gpu_memory $gpu_id; then
            run_on_gpu $gpu_id "--select-crit $selectcrit --few-shot $fewshot --gen-temp $gentemp --repeat $repeat"
            break # Break the loop if we found a GPU with enough memory
        fi
    done
    sleep 10
done

wait