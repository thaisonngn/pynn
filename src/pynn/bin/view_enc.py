#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import argparse
import matplotlib.pyplot as plt
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

parser.add_argument('--alg-beam', help='beam size', type=int, default=60)
parser.add_argument('--blank', help='blank field', type=int, default=2)
parser.add_argument('--blank-scale', help='blank scale', type=float, default=1.0)
parser.add_argument('--space', help='space token', type=str, default='▁')

def hide_axis(ax, label=None):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if label: ax.set_ylabel(label)

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
    with torch.no_grad():
        while True:
            seq, mask, utts = reader.read_batch_utt(1)
            if not utts: break
            seq, mask = seq.to(device), mask.to(device)
            utt = utts[0]
            hypo = [el+2 for el in lbl[utt]]

            probs, mask = model(seq, mask)
            probs = probs.softmax(dim=-1).squeeze(0).cpu()
            tgt = torch.LongTensor(hypo)
                        
            img = seq[0].cpu().numpy()
            hide_axis(plt.subplot(311))
            ax = plt.subplot(311)
            ax.get_yaxis().set_visible(False)
            ax.xaxis.tick_top()
            plt.imshow(img.T, aspect="auto")
        
            #alg = greedy_align(probs.log(), tgt)
            alg = viterbi_align(torch.log(probs), tgt, bs, blank, scale)
            if len(alg) != len(tgt):
                print(utt)
                print(tgt.tolist())
                print([a[0] for a in alg])
            probs = probs.transpose(1, 0)
            img = probs[tgt]
            hide_axis(plt.subplot(312))
            plt.imshow(img, aspect="auto")

            #print([a[0] for a in alg])
            #img = probs[tgt]
            for j, (tk, st, et) in enumerate(alg):
                #print('%s %d %d' % (dic[tk-2], st*4, et*4))
                #print('%s %d %d' % (tk, st*4, et*4))
                img[j][0:st] = 0.
                img[j][et:-1] = 0.
            hide_axis(plt.subplot(313))
            plt.imshow(img, aspect="auto")

            if dic is not None:
                hypo = [dic[token-2].replace(args.space, '') for token in hypo]
            plt.title(' '.join(map(str, hypo)), fontsize=10, y=-.4)
            #plt.imshow(img)

            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
                    
            plt.savefig('%s.png' % utt)
