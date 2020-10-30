#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1

export OUTPUT_DIR_NAME=outputs/${MODEL}_${RANDOM}
#export OUTPUT_DIR_NAME=outputs/${MODEL}
export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

rm -rf $OUTPUT_DIR

mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=10


export CUDA_VISIBLE_DEVICES=${GPUID}

python ${ROOT_DIR}/finetune.py \
--data_dir=${ROOT_DIR}/data/webnlg \
--learning_rate=3e-5 \
--num_train_epochs 100 \
--task graph2text \
--model_name_or_path=${MODEL} \
--train_batch_size=4 \
--eval_batch_size=4 \
--early_stopping_patience 15 \
--gpus 1 \
--output_dir=$OUTPUT_DIR \
--max_source_length=384 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_train --do_predict \
--eval_beams 3