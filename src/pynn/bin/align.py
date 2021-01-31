#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import os
import copy
import random
import numpy as np
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pynn.decoder.ctc import greedy_search, greedy_align, viterbi_align
from pynn.io.kaldi_seq import ScpStreamReader
from pynn.net.encoder import DeepLSTM

parser = argparse.ArgumentParser(description='pynn')

parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--source', help='path to source scp', required=True)
parser.add_argument('--target', help='label file', default=None)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--batch-size', help='batch size', type=int, default=40)

if __name__ == '__main__':
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.model_dic)
    model = DeepLSTM(**mdic['params']).to(device)
    model.load_state_dict(mdic['state'])
    model.eval()

    reader = ScpStreamReader(args.source, args.target, mean_sub=args.mean_sub, sek=False, downsample=args.downsample)
    reader.initialize()

    fout = open('label.lbl', 'w')
    with torch.no_grad():
        while True:
            src, mask, utts = reader.read_batch_utt(args.batch_size)
            if len(utts) == 0: break
            utt = utts[0]
            
            src, mask = src.to(device), mask.to(device)
            lbl = [[el+2 for el in reader.label_dic[utt]] for utt in utts]
            
            probs, mask = model(src, mask)
            probs = probs.log_softmax(dim=-1).cpu()
            lengths = mask.sum(-1).cpu()

            for utt, l, prob, lb in zip(utts, lengths, probs, lbl):
                prob, lb = prob[:l], torch.LongTensor(lb)
                alg = greedy_align(prob, lb)
                if len(alg) != len(lb):
                    alg = viterbi_align(prob, lb, 30, blank=2)
                if len(alg) != len(lb):
                    print(utt)
                    print(alg)
                    print(lb.tolist())
                tg = np.array([2] * l)
                for tk, st, et in alg: tg[st:et] = tk
                fout.write('%s %s\n' % (utt, ' '.join(map(str, tg))))
    fout.close()
