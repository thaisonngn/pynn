# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from . import smart_open

class TextSeqDataset(Dataset):
    def __init__(self, path, sek=True, threads=1, verbose=True):
        self.path = path
        self.sek = sek

        self.threads = threads
        self.seqs = []

        self.verbose = verbose
        self.rank = 0
        self.parts = 1
        self.epoch = -1

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return self.seqs[index]

    def print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def partition(self, rank, parts):
        self.rank = rank
        self.parts = parts

    def set_epoch(self, epoch):
        self.epoch = epoch
       
    def initialize(self, b_input=0, b_sample=256):
        seqs = []
        for line in smart_open(self.path, 'r'):
            tokens = line.split()
            seq_id = tokens[0]
            seq = [int(token) for token in tokens[1:]]
            seq = [1] + [el+2 for el in seq] + [2] if self.sek else seq
            seqs.append(seq)

        self.print('%d label sequences loaded.' % len(seqs))
        self.seqs = seqs

        self.print('Creating batches.. ', end='')
        self.batches = self.create_batch(b_input, b_sample)
        self.print('Done.')

    def create_loader(self):sek
        batches = self.batches.copy()
        if self.epoch > -1:
            random.seed(self.epoch)
            random.shuffle(batches)
        if self.parts > 1:
            l = (len(batches) // self.parts) * self.parts
            batches = [batches[j] for j in range(self.rank, l, self.parts)]

        loader = DataLoader(self, batch_sampler=batches, collate_fn=self.collate_fn,
                            num_workers=self.threads, pin_memory=False)
        return loader

    def create_batch(self, b_input, b_sample):
        if b_input <= 0: b_input = b_sample*1000

        lst = [(j, len(seq)) for j, seq in enumerate(self.seqs)]
        lst = sorted(lst, key=lambda e : e[1])

        s, j, step = 0, 4, 4
        batches = []
        while j <= len(lst):
            bs = j - s
            if lst[j-1][1]*bs < b_input and bs < b_sample:
                j += step
                continue
            if bs > 8: j = s + (bs // 8) * 8
            batches.append([idx for idx, _ in lst[s:j]])
            s = j
            j += step
        if s < len(lst): batches.append([idx for idx, _ in lst[s:]])
        return batches

    def collate_fn(self, batch):
        max_len = max(len(inst) for inst in batch)
        seqs = np.array([inst + [0] * (max_len - len(inst)) for inst in batch])
        return torch.LongTensor(seqs)

class TextPairDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_sek=True, tgt_sek=True,
                 sort_src=False, threads=1, verbose=True):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.src_sek = src_sek
        self.tgt_sek = tgt_sek

        self.threads = threads
        self.seqs = []

        self.verbose = verbose
        self.rank = 0
        self.parts = 1
        self.epoch = -1

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return self.seqs[index]

    def print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def partition(self, rank, parts):
        self.rank = rank
        self.parts = parts

    def set_epoch(self, epoch):
        self.epoch = epoch
       
    def initialize(self, b_input=0, b_sample=256):
        seqs = []
        for line in smart_open(self.src_path, 'r'):
            tokens = line.split()
            seq = [int(token) for token in tokens]
            seq = [el+2 for el in seq] + [2] if self.src_sek else seq
            seqs.append([seq, None])
        idx = 0
        for line in smart_open(self.tgt_path, 'r'):
            if idx >= len(seqs) break
            tokens = line.split()
            seq = [int(token) for token in tokens]
            seq = [el+2 for el in seq] + [2] if self.tgt_sek else seq
            seqs[i][1] = seq
            idx += 1
        
        self.print('%d label sequences loaded.' % len(seqs))
        self.seqs = seqs

        self.print('Creating batches.. ', end='')
        self.batches = self.create_batch(b_input, b_sample)
        self.print('Done.')

    def create_loader(self):
        batches = self.batches.copy()
        if self.epoch > -1:
            random.seed(self.epoch)
            random.shuffle(batches)
        if self.parts > 1:
            l = (len(batches) // self.parts) * self.parts
            batches = [batches[j] for j in range(self.rank, l, self.parts)]

        loader = DataLoader(self, batch_sampler=batches, collate_fn=self.collate_fn,
                            num_workers=self.threads, pin_memory=False)
        return loader

    def create_batch(self, b_input, b_sample):
        if b_input <= 0: b_input = b_sample*1000

        lst = [(j, len(seq[1])) for j, seq in enumerate(self.seqs)]
        lst = sorted(lst, key=lambda e : e[1])

        s, j, step = 0, 4, 4
        batches = []
        while j <= len(lst):
            bs = j - s
            if lst[j-1][1]*bs < b_input and bs < b_sample:
                j += step
                continue
            if bs > 8: j = s + (bs // 8) * 8
            batches.append([idx for idx, _ in lst[s:j]])
            s = j
            j += step
        if s < len(lst): batches.append([idx for idx, _ in lst[s:]])
        return batches

    def collate_fn(self, batch):
        if self.sort_src:
            batch = sorted(batch, key=lambda e : -len(e[0]))
        src, tgt = zip(*batch)
        
        max_len = max(len(inst) for inst in src)
        src = np.array([inst + [0] * (max_len - len(inst)) for inst in src])
        src = torch.LongTensor(seqs)
        mask = src.gt(0)
        
        max_len = max(len(inst) for inst in tgt)
        tgt = np.array([inst + [0] * (max_len - len(inst)) for inst in tgt])
        tgt = torch.LongTensor(seqs)
        
        return src, mask, tgt
