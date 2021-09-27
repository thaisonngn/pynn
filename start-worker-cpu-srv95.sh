#!/bin/bash

SYSTEM_PATH=`dirname "$0"`

export LD_LIBRARY_PATH=$SYSTEM_PATH/lib
export PYTHONPATH=$SYSTEM_PATH/pylib

#pythonCMD="/home/mtasr/anaconda3/envs/lt2021/bin/python -u -W ignore"
#pythonCMD="python -u -W ignore"
pythonCMD="/home/tnguyen/asr/anaconda3/envs/pytorch1.7/bin/python -u -W ignore"
log=logs/asr-en-srv95.worker.log

OMP_NUM_THREADS=8 $pythonCMD worker.py \
	--server "i13srv95.ira.uka.de" \
	--port 60019 \
	--name "asr-en" \
	--fingerprint "en-EU-memory$1" \
	--outfingerprint "en-EU" \
        --inputType "audio" \
        --outputType "text" \
        --dict "model/bpe4k.dic" \
        --model "model/s2s-lstm.dic" \
        --punct "model/punct.dic" \
        --device 'cpu' --beam-size 6 --new-words words$1.txt
