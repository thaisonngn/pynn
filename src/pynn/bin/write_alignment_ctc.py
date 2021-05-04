#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import argparse
import torch

from pynn.util import load_object
from pynn.decoder.ctc import beam_search
from pynn.decoder.ctc import greedy_search, greedy_align, viterbi_align
from pynn.util.text import load_dict, load_label
from pynn.io.audio_seq import SpectroDataset

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--lm-dic', help='language model dictionary', default=None)
parser.add_argument('--lm-scale', help='language model scale', type=float, default=0.5)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--data-lbl', help='label file', default=None)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--label', help='label file', default=None)

parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--alg-beam', help='beam size', type=int, default=20)
parser.add_argument('--blank', help='blank field', type=int, default=0)
parser.add_argument('--blank-scale', help='blank scale', type=float, default=1.0)
parser.add_argument('--pruning', help='pruning size', type=float, default=1.2)
parser.add_argument('--beam-size', help='beam size', type=int, default=12)
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

    lm = None
    if args.lm_dic is not None:
        mdic = torch.load(args.lm_dic)
        lm = load_object(mdic['class'], mdic['module'], mdic['params'])
        lm = lm.to(device)
        lm.load_state_dict(mdic['state'])
        lm.eval()
        if args.fp16: lm.half()

    reader = SpectroDataset(args.data_scp, mean_sub=args.mean_sub, downsample=args.downsample)
    bs, blank, scale = args.alg_beam, args.blank, args.blank_scale
    fout = open('label.lbl', 'w')
    with torch.no_grad():
        while True:
            src, mask, utts = reader.read_batch_utt(args.batch_size)
            if utts is None or len(utts) == 0: break

            src, mask = src.to(device), mask.to(device)
            
            if lbl is None:
                lbs = beam_search(model, src, mask, device, lm,
                                  args.lm_scale, args.beam_size, args.pruning, args.blank)
            else:
                lbs = [lbl[utt] for utt in utts]

            probs, mask = model(src, mask)[0:2]
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
                    token = dic[tk]
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

