#!/bin/bash

device="cuda"
GPUID="${1:--1}"
if [ "$GPUID" == "" ]; then
    echo "usage: bash start-worker-ELITR.sh GPUID [FILEID [WORKERID [SERVER [PORT]]]]"
    exit
elif [ "$GPUID" == "-1" ]; then
    device='cpu'
fi
FILEID="${2:-words1.txt}"
BEAMSIZE="${3:-4}"

SYSTEM_PATH=`dirname "$0"`

export LD_LIBRARY_PATH=$SYSTEM_PATH/lib
export PYTHONPATH=$SYSTEM_PATH/pylib

#pythonCMD="/home/chuber/miniconda3/envs/pytorch/python -u -W ignore"
pythonCMD="python -u -W ignore"

CUDA_VISIBLE_DEVICES=$GPUID OMP_NUM_THREADS=8 $pythonCMD worker.py \
        --dict "model/bpe4k.dic" \
        --model "model/s2s-lstm.dic" \
        --punct "model/punct.dic" \
        --device $device --beam-size $BEAMSIZE --new-words $FILEID
