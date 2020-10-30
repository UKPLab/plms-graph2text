#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1

export OUTPUT_DIR_NAME=outputs/${MODEL}_200k_${RANDOM}
#export OUTPUT_DIR_NAME=outputs/${MODEL}_test
export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

rm -rf $OUTPUT_DIR

mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=10


export CUDA_VISIBLE_DEVICES=${GPUID}
#--early_stopping_patience 15 \
python ${ROOT_DIR}/finetune.py \
--data_dir=${ROOT_DIR}/data/amr-200k \
--learning_rate=3e-5 \
--num_train_epochs 1 \
--task graph2text \
--model_name_or_path=${MODEL} \
--early_stopping_patience 15 \
--train_batch_size=4 \
--eval_batch_size=4 \
--gpus 1 \
--output_dir=$OUTPUT_DIR \
--max_source_length=384 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_train --do_predict \
--save_top_k 4 \
--eval_beams 5