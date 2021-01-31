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

from pynn.decoder.ctc import beam_search
from pynn.util import token2word, write_ctm, write_stm, write_text
from pynn.io.kaldi_seq import ScpStreamReader
from pynn.net.encoder import DeepLSTM
from pynn.net.lm import SeqLM
 
parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--lm-dic', help='language model dictionary', default=None)
parser.add_argument('--lm-scale', help='language model scale', type=float, default=0.1)

parser.add_argument('--dict', help='dictionary file', required=True)
parser.add_argument('--word-dict', help='word dictionary file', default=None)
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--beam-size', help='beam size', type=int, default=10)
parser.add_argument('--max-len', help='max len', type=int, default=100)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='hypos/H_1_LV.ctm')
parser.add_argument('--format', help='output format', type=str, default='ctm')
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

    lm = None
    if args.lm_dic is not None:
        mdic = torch.load(args.lm_dic)
        lm = SeqLM(**mdic['params']).to(device)
        lm.load_state_dict(mdic['state'])
        lm.eval()
        if args.fp16: lm.half()
    
    reader = ScpStreamReader(args.data_scp, mean_sub=args.mean_sub, fp16=args.fp16,
                             sort_src=True, sek=False, downsample=args.downsample)
    reader.initialize()

    since = time.time()
    beam_size, batch_size = args.beam_size, args.batch_size
    fout = open(args.output, 'w')
    while True:
        seq, mask, utts = reader.read_batch_utt(batch_size)
        if len(utts) == 0: break
        with torch.no_grad():
            seq, mask = seq.to(device), mask.to(device)
            preds = model.decode(seq, mask)
            hypos, scores = [], []
            for pred in preds:
                hypo = beam_search(pred, lm, args.lm_scale, beam_size=beam_size, topk=1, blank=2)[0]
                hypo = [el for el in hypo] + [2]
                #hypo = token2word(hypo, None, dic, word_dic, args.space)
                hypos.append(hypo)
                scores.append([1.0] * len(hypo))

            write_ctm(hypos, scores, fout, utts, dic, word_dic, args.space)

    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
