#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "./train_AMR.sh <model> <gpu_id>"
  exit 2
fi

#if [[ ${1} == *"bart-large"* ]]; then
#  bash amr/finetune_graph2text_bart_large.sh ${1} ${2}
#fi
if [[ ${1} == *"t5"* ]]; then
  bash amr/finetune_graph2text.sh ${1} ${2}
fi
if [[ ${1} == *"bart"* ]]; then
  bash amr/finetune_graph2text_bart.sh ${1} ${2}
fi








