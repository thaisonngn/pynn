#!/bin/bash

SYSTEM_PATH=`dirname "$0"`

export LD_LIBRARY_PATH=$SYSTEM_PATH/lib
export PYTHONPATH=$SYSTEM_PATH/pylib

pythonCMD="python -u -W ignore"
log=logs/asr-en-srv30.worker.log

OMP_NUM_THREADS=8 $pythonCMD worker.py \
	--server "i13srv30.ira.uka.de" \
	--port 60019 \
	--name "asr-en" \
	--fingerprint "en-EU" \
	--outfingerprint "en-EU" \
        --inputType "audio" \
        --outputType "text" \
        --dict "model/bpe4k.dic" \
        --model "model/s2s-lstm.dic" \
        --punct "model/punct.dic" \
        --device 'cpu' --int8 --beam-size 8

