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

from pynn.decoder.seq2seq import Beam, partial_search
from pynn.util import write_ctm
from pynn.io.kaldi_seq import ScpStreamReader
from pynn.net.seq2seq import Seq2Seq

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--lm-dic', help='language model dictionary', default=None)
parser.add_argument('--lm-scale', help='language model scale', type=float, default=0.5)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--word-dict', help='word dictionary file', default=None)
parser.add_argument('--data-scp', help='path to srouce scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--beam-size', help='beam size', type=int, default=8)
parser.add_argument('--attn-head', help='attention head', type=int, default=0)
parser.add_argument('--attn-padding', help='attention padding', type=float, default=0.05)
parser.add_argument('--stable-time', help='stable size', type=int, default=50)
parser.add_argument('--start-block', help='initlial block size', type=int, default=0)
parser.add_argument('--incl-block', help='incremental block size', type=int, default=50)
parser.add_argument('--max-len', help='max length', type=int, default=100)
parser.add_argument('--output', help='output file', type=str, default='hypos/H_1_LV.ctm')
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

    device = torch.device('cuda')
   
    mdic = torch.load(args.model_dic)
    model = Seq2Seq(**mdic['params']).to(device)
    model.load_state_dict(mdic['state'])
    model.eval()

    reader = ScpStreamReader(args.data_scp, mean_sub=args.mean_sub, downsample=args.downsample)
    reader.initialize()

    space, beam_size, max_len = args.space, args.beam_size, args.max_len
    start, win = args.start_block, args.incl_block
    stable_time, attn_padding = args.stable_time, args.attn_padding
    head = args.attn_head
    since = time.time() 
    fctm = open(args.output, 'w')    
    total_latency = 0
    count = 0    
    with torch.no_grad():
        while True:
            utt, mat = reader.read_next_utt()
            if utt is None or utt == '': break        
            print(utt)
            src_seq = torch.FloatTensor(mat).to(device)
            time_len = src_seq.size(0)
            stable_hypo = [1]
            latency = 0.

            for i in range(max(time_len-start-1, 0) // win + 1):
                end = min(time_len, win*(i+1)+start)
                src = src_seq[0:end, :]

                enc_out, mask, hypo, score, _ = partial_search(model, src, beam_size, max_len, stable_hypo)
                
                tgt = torch.LongTensor(hypo).to(device).view(1, -1)
                attn = model.get_attn(enc_out, mask, tgt)
                attn = attn[0]

                cs = torch.cumsum(attn[head], dim=1)
                ep = cs.le(1.-attn_padding).sum(dim=1)
                ep = ep.cpu().numpy()

                for j in range(len(stable_hypo), len(hypo)):
                    if ep[j-1]*4 + stable_time > end: break
                if j > len(stable_hypo):
                    sth = hypo[0:j]
                    latency += end * (len(sth) - len(stable_hypo))
                    stable_hypo = sth
                print(stable_hypo)

            latency += time_len * (len(hypo) - len(stable_hypo))
            latency /= time_len * (len(hypo)-1)
            count += 1; total_latency += latency
            print('Latency: %0.3f' % latency)

            write_ctm([hypo[1:]], [score[1:]], fctm, [utt], dic, word_dic, args.space)
    fctm.close()
    print('Final Latency: %0.3f' % (total_latency/count))
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))    
