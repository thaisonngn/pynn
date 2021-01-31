#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse
import numpy as np

from pynn.io.kaldi_io import write_ark_file
from pynn.io.audio_seq import SpectroDataset

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--inp-scp', help='path to input scp', required=True)
parser.add_argument('--out-ark', help='output ark', type=str, default='data.ark')
parser.add_argument('--out-scp', help='output scp', type=str, default='data.scp')
parser.add_argument('--max-len', help='maximum frames for a segment', type=int, default=10000)
parser.add_argument('--min-len', help='minimum frames for a segment', type=int, default=4)
parser.add_argument('--mean-norm', help='mean substraction', action='store_true')
parser.add_argument('--fp16', help='use float16 instead of float32', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    
    ds = SpectroDataset(args.inp_scp)
    ark_file = open(args.out_ark, 'wb')
    scp_file = open(args.out_scp, 'w')

    while True:
        utt, feats = ds.read_utt()
        if utt is None or utt == '': break    
        if len(feats) > args.max_len or len(feats) < args.min_len:
            continue
        if args.mean_norm:
            feats = feats - feats.mean(axis=0, keepdims=True)
        if args.fp16:
            feats = feats.astype(np.float16) 

        dic = {utt: feats}
        write_ark_file(ark_file, scp_file, dic)
    ark_file.close()
    scp_file.close()

