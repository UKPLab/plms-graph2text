#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "./preprocess_AMR.sh <dataset_folder>"
  exit 2
fi

bash amr/data/gen_LDC2017T10.sh ${1}




