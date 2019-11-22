#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

from pynn.io.kaldi_seq import ScpStreamReader

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--output', help='output file', type=str, default='data.len')

if __name__ == '__main__':
    args = parser.parse_args()
    
    loader = ScpStreamReader(args.data_scp)
    loader.initialize()

    fout = open(args.output, 'w')
    while True:
        utt_id, utt_mat = loader.read_next_utt()
        if utt_id is None or utt_id == '': break    
        fout.write(utt_id + u' ' + str(len(utt_mat)) + os.linesep)
    fout.close()
