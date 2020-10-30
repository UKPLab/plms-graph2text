#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "./decode_AMR.sh <model> <checkpoint> <gpu_id>"
  exit 2
fi

bash amr/test_graph2text.sh ${1} ${2} ${3}










