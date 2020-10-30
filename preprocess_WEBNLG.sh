#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "./preprocess_WEBNLG.sh <dataset_folder>"
  exit 2
fi

python webnlg/data/generate_input_webnlg.py ${1}




