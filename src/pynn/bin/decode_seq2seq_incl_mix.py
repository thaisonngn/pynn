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

from pynn.decoder.seq2seq import partial_search_multi
from pynn.util import write_ctm
from pynn.io.kaldi_seq import ScpStreamReader
from pynn.net.seq2seq import Seq2Seq
from pynn.net.tf import Transformer
from pynn.net.ensemble import Ensemble

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', type=str, action='append', nargs='+')
parser.add_argument('--smodel-dic', type=str, help='dictionary file', default=None)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--word-dict', help='word dictionary file', default=None)
parser.add_argument('--data-scp', help='path to srouce scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--beam-size', help='beam size', type=int, default=8)
parser.add_argument('--attn-head', help='attention head', type=int, default=0)
parser.add_argument('--attn-padding', help='attention padding', type=float, default=0.05)
parser.add_argument('--stable-time', help='stable size', type=int, default=50)
parser.add_argument('--prune', help='pruning threshold', type=float, default=1.0)
parser.add_argument('--start-block', help='initlial block size', type=int, default=0)
parser.add_argument('--incl-block', help='incremental block size', type=int, default=50)
parser.add_argument('--block-len', help='max block length', type=int, default=5)
parser.add_argument('--output', help='output file', type=str, default='hypos/H_1_LV.ctm')
parser.add_argument('--space', help='space token', type=str, default='<space>')
parser.add_argument('--fp16', help='float 16 bits', action='store_true')

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

    models = []
    for dic_path in args.model_dic:
        sdic = torch.load(dic_path[0])
        m_params = sdic['params']
        if sdic['type'] == 'lstm':
            model = Seq2Seq(**m_params).to(device)
        elif sdic['type'] == 'tf':
            model = Transformer(**m_params).to(device)
        model.load_state_dict(sdic['state'])
        model.eval()
        print(sum(p.numel() for p in model.parameters()))
        if args.fp16: model.half()
        models.append(model)
    model = Ensemble(models)
    if args.fp16: model.half()

    smodel = None
    if args.smodel_dic is not None:
        sdic = torch.load(args.smodel_dic)
        m_params = sdic['params']
        smodel = Seq2Seq(**m_params).to(device)
        if args.fp16: smodel.half() 
    reader = ScpStreamReader(args.data_scp, mean_sub=args.mean_sub, fp16=args.fp16, downsample=args.downsample)
    reader.initialize()

    space, beam_size, block_len = args.space, args.beam_size, args.block_len
    start, win, stable_time = args.start_block, args.incl_block, args.stable_time
    head, padding = args.attn_head, args.attn_padding
    
    since = time.time()
    fctm = open(args.output, 'w')
    total_latency, latency_count = 0, 0
    total_delay, total_rtf, rtf_count = 0., 0., 0
    with torch.no_grad():
        while True:
            utt, mat = reader.read_next_utt()
            if utt is None or utt == '': break        
            print(utt)
            src_seq = torch.HalfTensor(mat) if args.fp16 else torch.FloatTensor(mat)
            src_seq = src_seq.to(device)
            time_len = src_seq.size(0)
            stable_hypo = [1]
            latency = 0.

            for i in range(max(time_len-start-1, 0) // win + 1):
                end = min(time_len, win*(i+1)+start)
                src = src_seq[0:end, :]

                max_len = (i+1) * block_len - len(stable_hypo)
                stime = time.time()
                enc_out, mask, hypo, score, sth = partial_search_multi(model, src, beam_size, max_len, stable_hypo)
                rtf = (time.time() - stime) * 100 / src.size(0)
                rtf_count += 1; total_rtf += rtf; total_delay += time.time() - stime
                print('RTF: %0.3f' % rtf)
                print(sth)

                tgt = torch.LongTensor(hypo).to(device).view(1, -1)
                if smodel is None:
                    attn = model.get_attn(enc_out, mask, tgt)
                else:
                    mask = torch.ones((1, src.size(0)), dtype=torch.uint8).to(src.device)
                    enc_out, mask = smodel.encode(src.unsqueeze(0), mask)[0:2]
                    attn = smodel.get_attn(enc_out, mask, tgt)
                attn = attn[0]
                cs = torch.cumsum(attn[head], dim=1)
                ep = cs.le(1.-padding).sum(dim=1)
                ep = ep.cpu().numpy()

                if len(sth) > len(stable_hypo):
                    latency += end * (len(sth) - len(stable_hypo))
                    stable_hypo = sth

                for j in range(len(stable_hypo), len(hypo)):
                    if ep[j-1]*4 + stable_time > end: break
                if j > len(stable_hypo):
                    sth = hypo[0:j]
                    latency += end * (len(sth) - len(stable_hypo))
                    stable_hypo = sth
                    
            latency += time_len * (len(hypo) - len(stable_hypo))
            latency /= time_len * (len(hypo)-1)
            latency_count += 1; total_latency += latency
            print('Latency: %0.3f' % latency)
            
            write_ctm([hypo[1:]], [score[1:]], fctm, [utt], dic, word_dic, args.space)            
    fctm.close()
    print('Final Latency: %0.3f' % (total_latency/latency_count))
    print('Final RTF: %0.3f' % (total_rtf/rtf_count))
    print('Final Delay: %.3f' % (total_delay/rtf_count))
    time_elapsed = time.time() - since
    print("Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
