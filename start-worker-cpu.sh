#!/bin/bash

SYSTEM_PATH=`dirname "$0"`

export LD_LIBRARY_PATH=$SYSTEM_PATH/lib
export PYTHONPATH=$SYSTEM_PATH/pylib

SERVER="${1:-i13srv53.ira.uka.de}"
PORT="${2:-60019}"

#pythonCMD="python -u -W ignore"
pythonCMD="/home/mtasr/anaconda3/envs/lt2021/bin/python -u -W ignore"

OMP_NUM_THREADS=8 $pythonCMD worker.py \
        --server ${SERVER} \
        --port ${PORT} \
	--name "asr-en" \
	--fingerprint "en-EU" \
	--outfingerprint "en-EU" \
        --inputType "audio" \
        --outputType "text" \
        --dict "model/bpe4k.dic" \
        --model "model/s2s-lstm.dic" \
        --punct "model/punct.dic" \
        --device 'cpu' --beam-size 6

