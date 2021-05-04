#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import argparse
import torch

from pynn.util import load_object
from pynn.decoder.s2s import beam_search
from pynn.decoder.ctc import greedy_search, greedy_align, viterbi_align
from pynn.util.text import load_dict, load_label
from pynn.io.audio_seq import SpectroDataset

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--s2s-model', help='s2s model dictionary', default=None)
parser.add_argument('--model-dic', help='model dictionary', required=True)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--data-lbl', help='label file', default=None)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--label', help='label file', default=None)

parser.add_argument('--len-norm', help='length normalization', action='store_true')
parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--alg-beam', help='beam size', type=int, default=20)
parser.add_argument('--blank', help='blank field', type=int, default=0)
parser.add_argument('--blank-scale', help='blank scale', type=float, default=1.0)
parser.add_argument('--max-len', help='max len', type=int, default=200)
parser.add_argument('--space', help='space token', type=str, default='<space>')

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

    if args.s2s_model is not None:
        mdic = torch.load(args.s2s_model)
        s2s = load_object(mdic['class'], mdic['module'], mdic['params'])
        s2s.load_state_dict(mdic['state'])
        s2s = s2s.to(device)
        s2s.eval()
    else:
        s2s = model

    reader = SpectroDataset(args.data_scp, mean_sub=args.mean_sub, downsample=args.downsample)
    bs, blank, scale = args.alg_beam, args.blank, args.blank_scale
    fout = open('label.lbl', 'w')
    with torch.no_grad():
        while True:
            src, mask, utts = reader.read_batch_utt(args.batch_size)
            if utts is None or len(utts) == 0: break

            src, mask = src.to(device), mask.to(device)
            
            if lbl is None:
                hypos = beam_search(s2s, src, mask, device, 8, args.max_len, len_norm=args.len_norm)[0]
                lbs = []
                for hypo in hypos:
                    hp = []
                    for token in hypo:
                        if token == 2: break
                        hp.append(token)
                    lbs.append(hp)
            else:
                lbs = [[el+2 for el in lbl[utt]] for utt in utts]
            l = max(len(lb) for lb in lbs)
            tgt = [[1] + lb + [2] + [0]*(l-len(lb)) for lb in lbs]
            tgt = torch.LongTensor(tgt).to(device)

            mask, probs = model.align(s2s, src, mask, tgt)[1:3]
            probs = probs.log_softmax(dim=-1).cpu()
            lens = mask.sum(-1).cpu()

            for utt, l, prob, lb in zip(utts, lens, probs, lbs):
                #for tk in lb: print(dic[tk-2])            
                prob, lb = prob[:l], torch.LongTensor(lb)
                alg = viterbi_align(prob, lb, bs, blank, scale)
                if len(alg) != len(lb):
                    print(utt)
                    print(lb.tolist())
                    print([a[0] for a in alg])
                    continue
                #lb = lb.tolist()
                #wd = l*1. / len(lb)
                #alg = [(lb[j], int(j*wd)+1, int((j+1)*wd)) for j in range(len(lb))]
                txt = []
                word, ws, we = [], -1, -1
                for tk, st, et in alg:
                    token = dic[tk-2]
                    if ws == -1:
                        ws, we = st, et
                    if token.startswith(args.space):
                        if len(word) > 0:
                            txt.append('%s:%d:%d' % (''.join(word)[1:], ws*4, we*4))
                        word, ws = [token], st
                    else:
                        word.append(token)
                    we = et
                if len(word) > 0:
                    txt.append('%s:%d:%d' % (''.join(word)[1:], ws*4, we*4))
                fout.write('%s %s\n' % (utt, ' '.join(txt)))
    fout.close()

