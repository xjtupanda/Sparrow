VIDEO_TYPE="s,m,l"
# The names of the responsible persons for each pair of GPUs.

NAMES=(lyd jyg wzh wzz zcy by dyh lfy)


CKPT=$1
CKPT_FILE=$2
OUTPUT_DIR="bench_results/${CKPT}"

MODEL_PATH=${CKPT_FILE}

VIDEO_DIR="../../benchmarks/video-mme"
NUM_FRAMES=$3

# inference without sub-title
LOG=${OUTPUT_DIR}/eval_video-mme_${NUM_FRAMES}frame.log 
# Clear out the log file if it exists.
> "$LOG"
exec &> >(tee -a "$LOG")

for((i=0; i<${#NAMES[@]}; i++)); do
   CUDA_VISIBLE_DEVICES=${i} python -u video-mme_inference.py \
       --responsible_man ${NAMES[i]} \
       --model-path ${MODEL_PATH} \
       --video_type $VIDEO_TYPE \
       --num-frames ${NUM_FRAMES} \
       --video_dir ${VIDEO_DIR} \
       --output_path ${OUTPUT_DIR}/qa_${NUM_FRAMES}_wo_sub_revision &
done

wait

python normalize_video-mme.py \
    --result-dir ${OUTPUT_DIR}/qa_${NUM_FRAMES}_wo_sub_revision \
    --output-file ${OUTPUT_DIR}/video-mme/pred_merge.jsonl

GPULIST=(0 1 2 3 4 5 6 7)
CHUNKS=${#GPULIST[@]}


LOG_1=${OUTPUT_DIR}/eval_result_video-mme_${NUM_FRAMES}frame.log 
# Clear out the log file if it exists.
> "${LOG_1}"
exec &> >(tee -a "${LOG_1}")

for IDX in $(seq 0 $((CHUNKS-1))); do
   CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -u parse_answer_with_llm.py \
       --pred-file ${OUTPUT_DIR}/video-mme/pred_merge.jsonl \
       --output-file ${OUTPUT_DIR}/video-mme/results/${CHUNKS}_${IDX}.jsonl \
       --num-chunks $CHUNKS \
       --chunk-idx $IDX &
done

wait

output_file=${OUTPUT_DIR}/video-mme/results/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUTPUT_DIR}/video-mme/results/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval_vanilla.py \
    --result-file $output_file

