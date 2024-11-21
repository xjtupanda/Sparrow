#!/bin/bash


LOG=multi_task_training_log.txt


# Clear out the log file if it exists.
> "$LOG"
exec &> >(tee -a "$LOG")

bash finetune_abla3.sh
bash finetune_abla1.sh
bash finetune_abla2.sh
