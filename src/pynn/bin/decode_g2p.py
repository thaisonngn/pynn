#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import argparse

import torch

from pynn.util import load_object
from pynn.decoder.s2s import beam_search
from pynn.util.text import load_dict, write_hypo
 
parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--src-dict', help='src dictionary file', required=True)
parser.add_argument('--tgt-dict', help='tgt dictionary file', required=True)
parser.add_argument('--data-path', help='path to data file', required=True)

parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--beam-size', help='beam size', type=int, default=6)
parser.add_argument('--max-len', help='max len', type=int, default=40)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--len-norm', help='length normalization', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='hypos/output.text')

if __name__ == '__main__':
    args = parser.parse_args()

    src_dic = {}
    with open(args.src_dict, 'r') as f:
        for line in f:
            tokens = line.split()
            src_dic[tokens[0]] = int(tokens[1])
    tgt_dic = {}
    with open(args.tgt_dict, 'r') as f:
        for line in f: 
            tokens = line.split()
            tgt_dic[int(tokens[1])] = tokens[0]

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.model_dic)
    model = load_object(mdic['class'], mdic['module'], mdic['params'])
    model = model.to(device)
    model.load_state_dict(mdic['state'])
    model.eval()
    if args.fp16: model.half()
 
    data = []
    for line in open(args.data_path, 'r'):
        word = line.split()[0]
        word_norm = word[1:] if word.startswith('-') else word
        word_norm = word_norm[:-1] if word_norm.endswith('-') else word_norm
        seq = []
        skip = False
        for ch in word_norm:
            if ch not in src_dic:
                skip = True; break
            seq.append(src_dic[ch])
        if skip: continue
        #seq = [src_dic[ch] for ch in word]
        seq = [el+2 for el in seq] + [2]
        data.append((word, seq))

    bs = args.batch_size   
    since = time.time()
    fout = open(args.output, 'w')
    with torch.no_grad():
        start = 0
        while True:
            batch = data[start: start+bs]
            if len(batch) == 0: break
            words, seqs = zip(*batch)
            max_len = max(len(inst) for inst in seqs)
            src = [inst + [0] * (max_len - len(inst)) for inst in seqs]
            src = torch.LongTensor(src)
            mask = src.gt(0)

            src, mask = src.to(device), mask.to(device)
            hypos = beam_search(model, src, mask, device, args.beam_size, 
                                args.max_len, len_norm=args.len_norm)[0]
            for word, hypo in zip(words, hypos):
                j = 0
                while j < len(hypo) and hypo[j] != 2: j += 1
                hypo = [tgt_dic[tk-2] for tk in hypo[:j]]
                if len(hypo) == 0: continue
                fout.write(word + ' ' + ' '.join(hypo) + '\n')
            start += bs
    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
