#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import argparse

import numpy as np
import torch

from pynn.util import load_object
from pynn.decoder.ctc import beam_search
from pynn.util.text import load_dict, write_hypo
 
parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--src-dict', help='source dictionary', default=None)
parser.add_argument('--tgt-dict', help='target dictionary', default=None)

parser.add_argument('--data-path', help='path to data file', required=True)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='hypos/output.seg')
parser.add_argument('--space', help='space token', type=str, default='▁')

if __name__ == '__main__':
    args = parser.parse_args()

    dic, _ = load_dict(args.src_dict, None)

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
        tokens = line.split()
        seq = [int(token) for token in tokens]
        seq = [el+2 for el in seq] + [2]
        data.append(seq)

    puncts = {1:'', 2:'.', 3:',', 4:'?', 5:'!', 6:':', 7:';'}
    bs = 40                         
    since = time.time()
    fout = open(args.output, 'w')
    with torch.no_grad():
        start = 0
        while True:
            seqs = data[start: start+bs]
            if len(seqs) == 0: break
            max_len = max(len(inst) for inst in seqs)
            src = np.array([inst + [0] * (max_len - len(inst)) for inst in seqs])
            src = torch.LongTensor(src)
            mask = src.gt(0)
    
            src, mask = src.to(device), mask.to(device)
            logits = model(src, mask)[0]
            preds = torch.argmax(logits, -1).cpu().numpy()
            for seq, pred in zip(seqs, preds):
                hypo, tokens = [], []
                for j, el in enumerate(seq[:-1]):
                    token = dic[el-2]
                    if token.startswith(args.space) and len(tokens) > 0:
                        word, norm = ''.join(tokens), pred[j-1]-2
                        if norm > 7:
                            word = word.capitalize()
                            norm -= 7
                        if norm > 1:
                            word += puncts[norm]
                        hypo.append(word)
                        tokens = []
                    tokens.append(token[1:] if token.startswith(args.space) else token)

                if len(tokens) > 0:
                    word, norm = ''.join(tokens), pred[j]-2
                    if norm > 7:
                        word = word.capitalize()
                        norm -= 7
                    if norm > 1:
                        word += puncts[norm]
                    hypo.append(word)

                fout.write(' '.join(hypo) + '\n')
            start += bs
             
    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
