#!/bin/bash
data_scp=/export/data/test/eval2000-sorted.scp

pynndir=/home/user/pynn2020/src  # pointer to pynn

# export environment variables
export PYTHONPATH=$pynndir

pythonCMD="python -u -W ignore"

mkdir -p hypos
CUDA_VISIBLE_DEVICES=0 $pythonCMD $pynndir/pynn/bin/decode_s2s.py \
                                  --data-scp $data_scp --dict /export/data/bpe/m4k.dict \
                                  --model-dic "model/epoch-avg.dic" \
                                  --beam-size 8 --batch-size 64 --downsample 1 --mean-sub --fp16 --space '▁' 2>&1 | tee decode-s2s.log
