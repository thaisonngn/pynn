# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
import struct
import os
import numpy as np
import torch

from . import smart_open
 
class TextSeqReader(object):
    def __init__(self, path=None, sek=True, shuffle=False, fp16=False):
        self.path = path # path to the label file
        self.shuffle = shuffle
        self.sek = sek

        self.fp16 = fp16

        self.seq_arr= None

    def read_all_seq(self):
        self.seq_arr = []
        for line in smart_open(self.path, 'r'):
            tokens = line.split()
            seq_id = tokens[0]
            seq = [int(token) for token in tokens[1:]]
            seq = [1] + [el+2 for el in seq] + [2] if self.sek else seq
            self.seq_arr.append(seq)

    def initialize(self):
        if self.path is not None and self.seq_arr is None:
            self.read_all_seq()
            print("Loaded %d sequences" % len(self.seq_arr))
        
        if self.shuffle: random.shuffle(self.seq_arr)
                
        self.seq_index = 0
        self.seq_count = len(self.seq_arr)

    def available(self):
        return self.seq_index < self.seq_count

    def collate(self, seqs):
        max_len = max(len(inst) for inst in seqs)
        t_seqs = np.array([inst + [0] * (max_len - len(inst)) for inst in seqs])
        return torch.LongTensor(t_seqs)

    def next(self, batch_input=32):
        idx = self.seq_index
        seqs = self.seq_arr[idx : idx+batch_input]

        n_seq = len(seqs)
        self.seq_index += n_seq
         
        n_token = sum(len(inst) for inst in seqs)
        last = self.seq_index >= self.seq_count
        seqs = self.collate(seqs)
                
        return (seqs, n_seq, n_token, last)
