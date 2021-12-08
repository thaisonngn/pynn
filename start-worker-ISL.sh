#!/bin/bash

device="cuda"
GPUID="$1"
if [ "$GPUID" == "" ]; then
    echo "usage: bash start-worker-ELITR.sh GPUID [FILEID [WORKERID [SERVER [PORT]]]]"
    exit
elif [ "$GPUID" == "-1" ]; then
    device='cpu'
fi
FILEID="${2:-/project/OML/error_correction/data/spelled_words.txt}"
WORKERID="${3:-Stefan}"
SERVER="${4:-i13srv53.ira.uka.de}"
PORT="${5:-60019}"

SYSTEM_PATH=`dirname "$0"`

export LD_LIBRARY_PATH=$SYSTEM_PATH/lib
export PYTHONPATH=$SYSTEM_PATH/pylib


#pythonCMD="python -u -W ignore"
#pythonCMD="/home/mtasr/anaconda3/envs/lt2021/bin/python -u -W ignore"
pythonCMD="/home/tnguyen/asr/anaconda3/envs/pytorch1.7/bin/python -u -W ignore"

CUDA_VISIBLE_DEVICES=$GPUID OMP_NUM_THREADS=8 $pythonCMD worker.py \
        --server ${SERVER} \
        --port ${PORT} \
	--name "asr-en" \
	--fingerprint "en-EU-memory$WORKERID" \
	--outfingerprint "en-EU" \
        --inputType "audio" \
        --outputType "text" \
        --dict "model/bpe4k.dic" \
        --model "model/s2s-lstm.dic" \
        --punct "model/punct.dic" \
        --device $device --beam-size 4 --new-words $FILEID
