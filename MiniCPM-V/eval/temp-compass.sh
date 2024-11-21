#!/bin/bash


GPULIST=(0 1 2 3 4 5 6 7)
CHUNKS=${#GPULIST[@]}

CKPT=$1
CKPTFILE=$2
OUTPUT_DIR="bench_results/${CKPT}"

BASE_DIR="/data/pandayin/benchmarks/TempCompass"


#"/data/pandayin/MiniCPM-V/finetune/output/mix-data-few-samples-30k-finetune-tune-vision"
#"/data/pandayin/MiniCPM-V/finetune/output/full-video-finetune-tune-vision"
#"/data/pandayin/ckpt/MiniCPM-Llama3-V-2_5"
NUM_FRAMES=$3


LOG=${OUTPUT_DIR}/eval_TempCompass_${NUM_FRAMES}frame.log
# Clear out the log file if it exists.
> "$LOG"
exec &> >(tee -a "$LOG")


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m model_video_qa_loader \
        --model-path $CKPTFILE \
        --question-file ${BASE_DIR}/TempCompass-multi-choice.jsonl \
        --video-folder ${BASE_DIR} \
        --num-frames ${NUM_FRAMES} \
        --answers-file ${BASE_DIR}/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode mpt &
done

wait

output_file=${BASE_DIR}/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${BASE_DIR}/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


LOG_1=${OUTPUT_DIR}/eval_result_temp-compass_${NUM_FRAMES}frame.log 
# Clear out the log file if it exists.
> "${LOG_1}"
exec &> >(tee -a "${LOG_1}")

for IDX in $(seq 0 $((CHUNKS-1))); do
   CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -u parse_answer_with_llm.py \
       --pred-file "$output_file" \
       --output-file ${OUTPUT_DIR}/temp-compass/results/${CHUNKS}_${IDX}.jsonl \
       --num-chunks $CHUNKS \
       --chunk-idx $IDX &
done

wait

new_output_file=${OUTPUT_DIR}/temp-compass/results/merge.jsonl

# Clear out the output file if it exists.
> "$new_output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUTPUT_DIR}/temp-compass/results/${CHUNKS}_${IDX}.jsonl >> "$new_output_file"
done

python eval_vanilla.py \
    --result-file $new_output_file
