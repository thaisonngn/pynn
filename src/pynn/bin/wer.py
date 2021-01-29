#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

from pynn.util.text import levenshtein

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--source', help='hypothesis', required=True)
parser.add_argument('--target', help='stm reference', required=True)
parser.add_argument('--remove-suffix', help='skip time info suffix', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    
    source = {}
    for line in open(args.source, 'r'):
        tokens = line.split()
        uid, src = tokens[0], tokens[1:]
        if args.remove_suffix:
            uid = '-'.join(uid.split('-')[:-2])
        source[uid] = src
    errors, length, nseq, n10 = 0, 0, 0, 0
    for line in open(args.target, 'r'):
        tokens = line.split()
        uid, tgt = tokens[0], tokens[1:]
        if uid == '' or len(tgt) == 0: continue
        src = source.get(uid, [])
        sed = levenshtein(src, tgt)
        errors += sed
        length += len(tgt)
        wer = float(sed) / len(tgt)
        #print('%s: %.2f' % (uid, wer))
        if wer > 0.01: n10 += 1
        nseq += 1
    wer = float(errors) / length
    print('Overall WER: %.4f, Error Seq: %0.4f' % (wer*100, float(n10)/nseq))

