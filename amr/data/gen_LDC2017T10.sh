#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p ${ROOT_DIR}/data
REPO_DIR=${ROOT_DIR}/data/

DATA_DIR=${1}
mkdir -p ${REPO_DIR}/tmp_amr
PREPROC_DIR=${REPO_DIR}/tmp_amr
ORIG_AMR_DIR=${DATA_DIR}/data/amrs/split
mkdir -p ${REPO_DIR}/amr_ldc2017t10
FINAL_AMR_DIR=${REPO_DIR}/amr_ldc2017t10


mkdir -p ${PREPROC_DIR}/train
mkdir -p ${PREPROC_DIR}/dev
mkdir -p ${PREPROC_DIR}/test

mkdir -p ${FINAL_AMR_DIR}/train
mkdir -p ${FINAL_AMR_DIR}/dev
mkdir -p ${FINAL_AMR_DIR}/test


cat ${ORIG_AMR_DIR}/training/amr-* > ${PREPROC_DIR}/train/raw_amrs.txt
cat ${ORIG_AMR_DIR}/dev/amr-* > ${PREPROC_DIR}/dev/raw_amrs.txt
cat ${ORIG_AMR_DIR}/test/amr-* > ${PREPROC_DIR}/test/raw_amrs.txt

mkdir -p ${ROOT_DIR}/amr17

for SPLIT in train dev test; do
    echo "processing $SPLIT..."
    python ${ROOT_DIR}/split_amr.py ${PREPROC_DIR}/${SPLIT}/raw_amrs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${PREPROC_DIR}/${SPLIT}/graphs.txt
    python ${ROOT_DIR}/preproc_amr.py ${PREPROC_DIR}/${SPLIT}/graphs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${FINAL_AMR_DIR}/${SPLIT}/nodes.pp.txt ${FINAL_AMR_DIR}/${SPLIT}/surface.pp.txt --mode LIN --triples-output ${FINAL_AMR_DIR}/${SPLIT}/triples.pp.txt

    cp ${FINAL_AMR_DIR}/${SPLIT}/nodes.pp.txt ${ROOT_DIR}/amr17/${SPLIT}.source
    cp ${FINAL_AMR_DIR}/${SPLIT}/surface.pp.txt ${ROOT_DIR}/amr17/${SPLIT}.target
    perl ${ROOT_DIR}/tokenizer.perl -l en < ${ROOT_DIR}/amr17/${SPLIT}.target > ${ROOT_DIR}/amr17/${SPLIT}.target.tok

    echo "done."

done

mv ${ROOT_DIR}/amr17/dev.source ${ROOT_DIR}/amr17/val.source
mv ${ROOT_DIR}/amr17/dev.target ${ROOT_DIR}/amr17/val.target
mv ${ROOT_DIR}/amr17/dev.target.tok ${ROOT_DIR}/amr17/val.target.tok

rm -rf ${REPO_DIR}


