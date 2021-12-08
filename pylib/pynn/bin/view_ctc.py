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

from pynn.decoder.ctc import greedy_search, align_label
from pynn.io.kaldi_seq import ScpStreamReader
from pynn.net.rnn_ctc import DeepLSTM
 
parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)

parser.add_argument('--dict', help='dictionary file', required=True)
parser.add_argument('--word-dict', help='word dictionary file', default=None)
parser.add_argument('--source', help='path to data scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--beam-size', help='beam size', type=int, default=10)
parser.add_argument('--max-len', help='max len', type=int, default=100)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--space', help='space token', type=str, default='<space>')

if __name__ == '__main__':
    args = parser.parse_args()

    dic = None
    if args.dict is not None:
        dic = {}
        fin = open(args.dict, 'r')
        for line in fin:
            tokens = line.split()
            dic[int(tokens[1])] = tokens[0]
    word_dic = None
    if args.word_dict is not None:
        fin = open(args.word_dict, 'r')
        word_dic = {}
        for line in fin:
            tokens = line.split()
            word_dic[''.join(tokens[1:])] = tokens[0]

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.model_dic)
    model = DeepLSTM(**mdic['params']).to(device)
    model.load_state_dict(mdic['state'])
    model.eval()
    if args.fp16: model.half()
    
    reader = ScpStreamReader(args.source, mean_sub=args.mean_sub, fp16=args.fp16,
                             sort_src=True, sek=False, downsample=args.downsample)
    reader.initialize()

    with torch.no_grad():
        while True:
            src, mask, utts = reader.read_batch_utt(1)
            if len(utts) == 0: break
            utt = utts[0]
            
            src, mask = src.to(device), mask.to(device)
            probs = model(src, mask)[0].cpu()
            tgt = greedy_search(probs)

            fig = plt.figure()
            probs, tgt = probs.squeeze(0), tgt[0]
            alg = align_label(probs, tgt)
            print(tgt)
            probs = probs.transpose(1, 0)
            img = probs[tgt]
            plt.subplot(211)
            plt.imshow(img)

            for j, (tk, st, et) in enumerate(alg):
                print('%s %d %d' % (dic[tk], st*4, et*4))
                img[j][0:st] = 0.
                img[j][et:-1] = 0.
            plt.subplot(212)
            plt.imshow(img)


            if dic is not None:
                hypo = [dic[token].replace(args.space, '') for token in tgt]
            plt.title(' '.join(map(str, hypo)), fontsize=8)
            #plt.imshow(img)

            plt.savefig('%s.png' % utt)
