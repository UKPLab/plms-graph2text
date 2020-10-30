#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "./preprocess_AGENDA.sh <dataset_folder>"
  exit 2
fi

python agenda/data/generate_input_agenda.py ${1}