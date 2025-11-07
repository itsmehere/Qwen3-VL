#!/bin/bash

# Get the project root directory (parent of qwen-vl-finetune)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# GPU configuration - specify directly in script
NPROC_PER_NODE=4
CUDA_VISIBLE_DEVICES=0,1,2,3

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen3-VL-4B-Instruct

# Training hyperparameters
lr=5e-5
batch_size=2
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets="rlbench_icl_8_full_train,real_icl_8_full_train"
eval_datasets="rlbench_icl_8_full_val,real_icl_8_full_val"

# Output configuration
wandb_project="Qwen3-ICL"
run_name="Qwen3-4B-Trace-RLBench-Real"
output_dir=./models/Qwen3-4B-Trace-RLBench-Real

# Create output directory and save script copy
mkdir -p ${output_dir}
timestamp=$(date +"%Y%m%d_%H%M%S")
script_copy="${output_dir}/sft_script_${timestamp}.sh"
cp "$0" "${script_copy}"
echo "Saved script copy to: ${script_copy}"

# Training arguments
args="
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 5.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "steps" \
    --eval_steps 250 \
    --eval_dataset_use ${eval_datasets} \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Report to team
export WANDB_PROJECT="${wandb_project}"
export WANDB_API_KEY="f437168ada63e2a37a8ede24bdff617daf8cdee6"

# Launch training
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}