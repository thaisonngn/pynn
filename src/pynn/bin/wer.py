#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

from pynn.util.decoder import levenshtein

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--hypo', help='hypothesis', required=True)
parser.add_argument('--ref', help='stm reference', required=True)
parser.add_argument('--ref-field', help='stm reference', type=int, default=4)

if __name__ == '__main__':
    args = parser.parse_args()
    
    hypos = {}
    with open(args.hypo, 'r') as f:
        for line in f:
            tokens = line.split()
            uid, hypo = tokens[0], tokens[1:]
            hypos[uid] = hypo
    rf = args.ref_field
    err = 0
    l = 0
    n = 0
    n10 = 0
    with open(args.ref, 'r') as f:
        for line in f:
            tokens = line.split()
            uid = tokens[0]
            ref = tokens[rf:]
            hypo = hypos[uid]
            sed = levenshtein(hypo, ref)
            wer = float(sed) / len(ref)
            print('%s: %.2f' % (uid, wer))
            err += sed
            l += len(ref)
            if wer > 0.1: n10 += 1
            n += 1
    wer = float(err) / l
    print('Overall WER: %.4f, Error Utter: %0.4f' % (wer, float(n10)/n))

