#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "./train_WEBNLG.sh <model> <gpu_id>"
  exit 2
fi

if [[ ${1} == *"t5"* ]]; then
  bash webnlg/finetune_graph2text.sh ${1} ${2}
fi
if [[ ${1} == *"bart"* ]]; then
  bash webnlg/finetune_graph2text_bart.sh ${1} ${2}
fi








