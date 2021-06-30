#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import argparse
import numpy as np
import torch

from pynn.util import load_object
from pynn.decoder.s2s import beam_search
from pynn.decoder.ctc import greedy_search, greedy_align, viterbi_align
from pynn.util.text import load_dict, load_label
from pynn.io.audio_seq import SpectroDataset

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--data-lbl', help='label file', default=None)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--label', help='label file', default=None)

parser.add_argument('--batch-size', help='batch size', type=int, default=80)
parser.add_argument('--alg-beam', help='beam size', type=int, default=60)
parser.add_argument('--blank', help='blank field', type=int, default=2)
parser.add_argument('--blank-scale', help='blank scale', type=float, default=1.0)

if __name__ == '__main__':
    args = parser.parse_args()

    dic = load_dict(args.dict)[0]
    lbl = None if args.data_lbl is None else load_label(args.data_lbl)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.model_dic)
    model = load_object(mdic['class'], mdic['module'], mdic['params'])
    model = model.to(device)
    model.load_state_dict(mdic['state'])
    model.eval()

    reader = SpectroDataset(args.data_scp, mean_sub=args.mean_sub, downsample=args.downsample)
    bs, blank, scale = args.alg_beam, args.blank, args.blank_scale
    fout = open('label.lbl', 'w')
    with torch.no_grad():
        while True:
            src, mask, utts = reader.read_batch_utt(args.batch_size)
            if utts is None or len(utts) == 0: break
            src, mask = src.to(device), mask.to(device)
            lbs = [[el+2 for el in lbl[utt]] for utt in utts]

            probs, mask = model(src, mask)
            probs = probs.log_softmax(dim=-1).cpu()
            lens = mask.sum(-1).cpu()

            for utt, l, prob, lb in zip(utts, lens, probs, lbs):
                prob, lb = prob[:l], torch.LongTensor(lb)
                #alg = greedy_align(prob, lb)
                #if len(alg) != len(lb):
                alg = viterbi_align(prob, lb, bs, blank, scale)
                if len(alg) != len(lb):
                    print(utt)
                    print(lb.tolist())
                    print([a[0] for a in alg])
                    continue
                tg = np.array([2] * l)
                for tk, st, et in alg: tg[st:et] = tk
                fout.write('%s %s\n' % (utt, ' '.join(map(str, tg))))
    fout.close()

