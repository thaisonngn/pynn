#!/bin/bash
data_dir=/export/data/swb-v15

pynndir=/home/user/pynn2020/src  # pointer to pynn

# export environment variables
export PYTHONPATH=$pynndir
export OMP_NUM_THREADS=4

pythonCMD="python -u -W ignore"

mkdir -p model

CUDA_VISIBLE_DEVICES=0 $pythonCMD $pynndir/pynn/bin/train_s2s_lstm.py \
                                  --train-scp $data_dir/tr.scp --train-target $data_dir/tr-label.bpe1k \
                                  --valid-scp $data_dir/cv.scp --valid-target $data_dir/cv-label.bpe1k \
                                  --n-classes 1003 --d-input 40 --d-enc 1024 --n-enc 6 \
                                  --use-cnn --freq-kn 3 --freq-std 2 --downsample 1 --mean-sub \
                                  --d-dec 1024 --n-dec 2 --d-emb 512 --d-project 300 --n-head 1 \
                                  --enc-dropout 0.3 --enc-dropconnect 0.3 --dec-dropout 0.2 --dec-dropconnect 0.2 --emb-drop 0.15 \
                                  --teacher-force 0.8 --weight-decay 0.000006 --weight-noise \
                                  --spec-drop --spec-bar 6 --spec-ratio 0.4 --time-stretch --fp16 \
				  --n-epoch 100 --lr 0.002 --b-input 40000 --b-update 8000 --n-warmup 8000 --n-const 4000 2>&1 | tee run-seq2seq.log
