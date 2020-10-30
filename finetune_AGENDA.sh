#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "./train_AGENDA.sh <model> <gpu_id>"
  exit 2
fi

bash agenda/finetune_graph2text.sh ${1} ${2}







