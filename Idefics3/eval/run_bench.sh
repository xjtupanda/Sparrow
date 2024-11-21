#!/bin/bash


CKPT=$1
CKPT_FILE=$2
NUM_FRAMES=$3
OUTPUT_DIR="bench_results/${CKPT}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

LOG=${OUTPUT_DIR}/multi-bench-evaluation_log.txt 
#
# Clear out the log file if it exists.
> "$LOG"
exec &> >(tee -a "$LOG")

bash yt_video_inference.sh ${CKPT} ${CKPT_FILE} ${NUM_FRAMES}
#bash mvbench.sh ${CKPT} ${CKPT_FILE}
#bash temp-compass.sh ${CKPT} ${CKPT_FILE}
#bash vcgbench.sh ${CKPT} ${CKPT_FILE}
