#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import argparse

import torch

from pynn.util import load_object
from pynn.decoder.ctc import beam_search
from pynn.util.text import load_dict, write_hypo
from pynn.io.audio_seq import SpectroDataset
 
parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--lm-dic', help='language model dictionary', default=None)
parser.add_argument('--lm-scale', help='language model scale', type=float, default=0.5)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--word-dict', help='word dictionary file', default=None)
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--blank', help='blank', type=int, default=0)
parser.add_argument('--beam-size', help='beam size', type=int, default=10)
parser.add_argument('--pruning', help='pruning size', type=float, default=1.5)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='hypos/H_1_LV.ctm')
parser.add_argument('--format', help='output format', type=str, default='ctm')
parser.add_argument('--space', help='space token', type=str, default='<space>')

if __name__ == '__main__':
    args = parser.parse_args()

    dic, word_dic = load_dict(args.dict, args.word_dict)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.model_dic)
    model = load_object(mdic['class'], mdic['module'], mdic['params'])
    model = model.to(device)
    model.load_state_dict(mdic['state'])
    model.eval()
    if args.fp16: model.half()
 
    lm = None
    if args.lm_dic is not None:
        mdic = torch.load(args.lm_dic)
        lm = load_object(mdic['class'], mdic['module'], mdic['params'])
        lm = lm.to(device)
        lm.load_state_dict(mdic['state'])
        lm.eval()
        if args.fp16: lm.half()
        
    reader = SpectroDataset(args.data_scp, mean_sub=args.mean_sub, fp16=args.fp16,
                            sort_src=True, sek=False, downsample=args.downsample)
    since = time.time()
    fout = open(args.output, 'w')
    with torch.no_grad():
        while True:
            seqs, masks, utts = reader.read_batch_utt(args.batch_size)
            if not utts: break
            seqs, masks = seqs.to(device), masks.to(device)
            hypos = beam_search(model, seqs, masks, device, lm,
                                args.lm_scale, args.beam_size, args.pruning, args.blank)
            hypos = [[el+2-args.blank for el in hypo] + [2] for hypo in hypos]
             
            write_hypo(hypos, None, fout, utts, dic, word_dic, args.space, args.format)
    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
