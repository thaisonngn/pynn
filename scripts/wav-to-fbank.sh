#!/bin/bash
audio_dir=/export/data/tedlium_v3/wav

pynndir=/home/user/pynn2020  # pointer to pynn

# export environment variables
export PYTHONPATH=$pynndir
export OMP_NUM_THREADS=2

pythonCMD="python -u -W ignore"
mkdir -p data
$pythonCMD $pynndir/pynn/bin/wav_to_fbank.py --seg-desc $1 --max-len 5000 --min-len 10 --jobs 10 --fp16 --wav-path $audio_dir --output data/data
