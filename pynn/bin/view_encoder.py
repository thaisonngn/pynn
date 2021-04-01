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

import matplotlib.pyplot as plt

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
parser.add_argument('--dict', help='dictionary file', default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    dic = None
    if args.dict is not None:
        dic = {}
        for line in open(args.dict, 'r'):
            tokens = line.split()
            dic[int(tokens[1])] = tokens[0]

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
            src, mask, utts = reader.read_batch_utt(1)
            if len(utts) == 0: break
            utt = utts[0]
            
            src, mask = src.to(device), mask.to(device)
            lbl = [el+2 for el in reader.label_dic[utt]]
            
            probs, mask = model(src, mask)
            probs = probs.softmax(dim=-1).cpu()

            fig = plt.figure()
            prob, lb = probs.squeeze(0), torch.LongTensor(lbl)
            print(lbl)
            alg = greedy_align(prob.log(), lb)
            print(alg)
            #if len(alg) != len(lbl):
            alg = viterbi_align(prob.log(), lb, 10, blank=2)
            print(alg)
            prob = prob.transpose(1, 0)
            img = prob[lb]
            plt.subplot(211)
            plt.imshow(img)

            for j, (tk, st, et) in enumerate(alg):
                print('%s %d %d' % (dic[tk-2], st, et))
                img[j][0:st] = 0.
                img[j][et:-1] = 0.
            plt.subplot(212)
            plt.imshow(img)

            plt.savefig('%s.png' % utt)

    fout.close()
