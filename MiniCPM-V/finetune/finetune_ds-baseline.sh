#!/bin/bash

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=9001

OUTPUT_DIR='output/full-video-finetune-tune-vision'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

LOG=${OUTPUT_DIR}/training_log_node.txt


# Clear out the log file if it exists.
> "$LOG"
exec &> >(tee -a "$LOG")
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

MODEL="/data/pandayin/ckpt/MiniCPM-Llama3-V-2_5"
# or openbmb/MiniCPM-V-2, openbmb/MiniCPM-Llama3-V-2_5
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/data/pandayin/data/minicpm-full-video-24frames-200k.json"
#EVAL_DATA="path/to/test_data"
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3"
MODEL_MAX_Length=8666 # if conduct multi-images sft, please set MODEL_MAX_Length=4096


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
#--eval_data_path $EVAL_DATA \
#--eval_steps 1000 \
# --per_device_eval_batch_size 1 \
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --label_names "labels" \
    --remove_unused_columns false \
    --bf16 true \
    --eval_strategy "no" \
    --do_train \
    --do_eval false \
    --tune_vision true \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 1 \
    --num_train_epochs 1 \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 9999 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero3.json \
    --report_to none 
