# Copyright 2019 Christian Huber
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import pickle
import sentencepiece as spm

from tqdm import tqdm
import time

class MemoryDataset(Dataset):
    def __init__(self, dataset, args, n_sim=[1,2,3], validation=False, fast=False):
        self.dataset = dataset

        self.n_sim = n_sim
        self.validation = validation
        self.fast = fast

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(args.bpe_model)

        self.size_memory = args.n_memory
        self.allowed_start_ids = [i+2 for i in range(args.n_classes-3) if self.sp.id_to_piece(i)[0]=="▁"]

        if self.validation:
            random.seed(42)
            if not fast:
                self.indices = [self.sample_random(indices) for indices in tqdm(self.dataset.batches)]

    def encode_as_ids(self, label):
        return [1] + [el + 2 for el in self.sp.encode_as_ids(label)] + [2]

    def decode_ids(self, ids):
        return self.sp.decode_ids([i-2 for i in ids[1:-1] if i>=3])

    def sample_random(self, indices):
        labels = [self.dataset.utt_lbl[i][-1] for i in indices]

        ind_to_stats = {}
        for _ in range(min(self.size_memory,2*len(labels))):
            for _ in range(100):
                n_sim = random.choice(self.n_sim)

                y = random.randint(0,len(labels)-1)
                label = labels[y]
                if len(label) <= 2:
                    continue
                x = random.randint(1,len(label)-2)
                if not label[x] in self.allowed_start_ids:
                    continue
                anz = 0
                b = False
                for x2 in range(x+1,len(label)):
                    if label[x2] in self.allowed_start_ids or label[x2]==2:
                        anz += 1
                        if anz == n_sim:
                            if not y in ind_to_stats:
                                ind_to_stats[y] = [[x,x2]]
                                b = True
                            else:
                                b2 = False
                                for x_,x2_ in ind_to_stats[y]:
                                    if x_<x2<=x2_ or x_<=x<x2_:
                                        b2 = True
                                        break
                                if b2:
                                    break
                                ind_to_stats[y].append([x,x2])
                                b = True
                            break
                if b:
                    break

        tgt_ids_mem = [[1] + labels[y][x:x2] + [2] for y,v in ind_to_stats.items() for x,x2 in v]

        # fill memory to size
        indices2 = []
        tgt_ids_mem2 = []
        while len(tgt_ids_mem)+len(indices2) < self.size_memory:
            i = random.randint(1,len(self.dataset)-2)
            if not i in indices and not i in indices2:
                for _ in range(100):
                    n_sim = random.choice(self.n_sim)

                    label = self.dataset.utt_lbl[i][-1]
                    if len(label) <= 2:
                        continue
                    x = random.randint(0,len(label)-1)
                    if not label[x] in self.allowed_start_ids:
                        continue
                    anz = 0
                    b = False
                    for x2 in range(x+1, len(label)):
                        if label[x2] in self.allowed_start_ids or label[x2]==2:
                            anz += 1
                            if anz == n_sim:
                                indices2.append(i)
                                tgt_ids_mem2.append([1] + label[x:x2] + [2])
                                b = True
                                break
                    if b:
                        break

        return tgt_ids_mem+tgt_ids_mem2, ind_to_stats

    def shuffle(self):
        self.reordering = random.sample(range(len(self.dataset.batches)),len(self.dataset.batches))

    def get_similarity_mask(self, tgt, ind_to_stats):
        mask = torch.zeros((tgt.shape[0], tgt.shape[1] - 1), dtype=torch.int64)

        index = 1
        for y,v in ind_to_stats.items():
            for x,x2 in v:
                mask[y,x-1:x2-1]= index
                index += 1
        return mask

    def __getitem__(self, i):
        if not self.validation:
            i = self.reordering[i]

        indices = self.dataset.batches[i]
        tmp = [self.dataset[i] for i in indices]

        if self.validation and not self.fast:
            tgt_ids_mem, ind_to_stats = self.indices[i]
        else:
            tgt_ids_mem, ind_to_stats = self.sample_random(indices)

        src_seq, src_mask, tgt_seq = self.dataset.collate_fn(tmp)
        tgt_ids_mem = self.dataset.collate_tgt(tgt_ids_mem)[0]
        label_mem = self.get_similarity_mask(tgt_seq, ind_to_stats)

        """for y in range(tgt_seq.shape[0]):
            print(tgt_seq[y])
            print(self.decode_ids([int(x) for x in tgt_seq[y]]))
            print(mask_sim[y])
            ids = set([int(x) for x in list(mask_sim[y]) if not int(x)==0])
            print(ids)
            for i in ids:
                print(tgt_ids_mem[i-1])
                print(self.decode_ids([int(x) for x in tgt_ids_mem[i-1]]))
        sys.exit()"""

        return src_seq, src_mask, tgt_seq, tgt_ids_mem, label_mem

    def __len__(self):
        return len(self.dataset.batches)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def create_loader(self):
        return DataLoader(self, batch_size=None, num_workers=16, pin_memory=False)




























