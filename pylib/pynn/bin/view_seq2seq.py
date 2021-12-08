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

from pynn.decoder.seq2seq import beam_search
from pynn.io.kaldi_seq import ScpStreamReader
from pynn.net.seq2seq import Seq2Seq
 
parser = argparse.ArgumentParser(description='pynn')

parser.add_argument('--model-dic', help='model dictionary', required=True)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--source', help='path to srouce scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--len-norm', help='length normalization', action='store_true')

parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--space', help='space token', type=str, default='<space>')
parser.add_argument('--beam-size', help='beam size', type=int, default=8)
parser.add_argument('--max-len', help='max len', type=int, default=200)

parser.add_argument('--head', help='head number', type=int, default=0)
parser.add_argument('--win', help='max len', type=int, default=10000)

if __name__ == '__main__':
    args = parser.parse_args()

    dic = None
    if args.dict is not None:
        dic = {}
        fin = open(args.dict, 'r')
        for line in fin:
            tokens = line.split()
            dic[int(tokens[1])] = tokens[0]

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.model_dic)
    model = Seq2Seq(**mdic['params']).to(device)
    model.load_state_dict(mdic['state'])
    model.eval()
    
    reader = ScpStreamReader(args.source, mean_sub=args.mean_sub, sort_src=True, downsample=args.downsample)
    reader.initialize()

    space, beam_size, max_len = args.space, args.beam_size, args.max_len
    win = args.win

    with torch.no_grad():
        while True:
            #src_seq, src_mask, tgt_seq = reader.next_batch(1)
            src_seq, src_mask, utts = reader.read_batch_utt(1)
            if len(utts) == 0: break
            utt = utts[0]
            
            
            fout = open(utt+'.info', 'w')
            time_len = src_seq.size(1)
            for i in range((time_len-1) // win + 1):
                e = min(time_len, win*(i+1))
                src = src_seq[:, 0:e, :]
                mask = src_mask[:, 0:e]
                
                src, mask = src.to(device), mask.to(device)
                hypos, scores = beam_search(model, src, mask, device, beam_size, max_len, len_norm=args.len_norm)
                hypo, score = [], []
                for token, s in zip(hypos[0], scores[0]):
                    if token == 2: break
                    hypo.append(token)
                    score.append(s)
                print(hypo)
                                
                tgt = torch.LongTensor([1]+hypo+[2]).to(device).view(1, -1)
                attn = model.attend(src, mask, tgt)[1]
                attn = attn[0]

                #print(attn)
                cs = torch.cumsum(attn[args.head], dim=1)
                sp = cs.le(0.05).sum(dim=1)
                ep = cs.le(0.95).sum(dim=1)

                sp, ep = sp.cpu().numpy(), ep.cpu().numpy()
                
                fout.write("# End=%d\n" % (e//4))
                for j, token in enumerate(hypo):
                    js, je = sp[j], ep[j]
                    fout.write("%s: %d %d  %f\n" % (dic[token-2], js, je, math.exp(score[j])))
                fout.write("\n")
                
                attn = attn.view(-1, attn.size(2))
                attn = attn.cpu().numpy()
                fig = plt.figure()
                plt.imshow(attn)

                if dic is not None:
                    hypo = [dic[token-2].replace(space, '') for token in hypo]
                plt.title(' '.join(map(str, hypo)))
                
                plt.savefig('%s-%d.png' % (utt, i))
            
            fout.close()
