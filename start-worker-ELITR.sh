#!/bin/bash

device="cuda"
GPUID="$1"
if [ "$GPUID" == "" ]; then
    echo "usage: bash start-worker-ELITR.sh GPUID [FILEID]"
    exit
elif [ "$GPUID" == "-1" ]; then
    device='cpu'
fi
FILEID="${2:-ID}"

SYSTEM_PATH=`dirname "$0"`

export LD_LIBRARY_PATH=$SYSTEM_PATH/lib
export PYTHONPATH=$SYSTEM_PATH/pylib

#pythonCMD="python -u -W ignore"
pythonCMD="/home/tnguyen/asr/anaconda3/envs/pytorch1.7/bin/python -u -W ignore"
log=logs/asr-en-mediator1.worker.log

CUDA_VISIBLE_DEVICES=$GPUID OMP_NUM_THREADS=8 $pythonCMD worker.py \
	--server "localhost" \
	--port 60020 \
	--name "asr-en-memory" \
	--fingerprint "en-EU-memory$FILEID" \
	--outfingerprint "en-EU" \
        --inputType "audio" \
        --outputType "text" \
        --dict "model/bpe4k.dic" \
        --model "model/s2s-lstm.dic" \
        --punct "model/punct.dic" \
        --device $device --beam-size 4 --new-words "http://ufallab.ms.mff.cuni.cz/~kumar/elitr/live-adaptation/memory-$FILEID.txt" 
