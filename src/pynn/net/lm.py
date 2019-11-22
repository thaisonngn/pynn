# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqLM(nn.Module):
    def __init__(self, output_size, hidden_size, layers, bn_size=0, shared_emb=True,
            dropout=0.2, emb_drop=0.):
        super().__init__()

        # Define layers
        self.emb = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.scale = hidden_size**0.5 if shared_emb else 1.
        self.emb_drop = nn.Dropout(emb_drop)

        dropout = (0 if layers == 1 else dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, layers, dropout=dropout, batch_first=True)

        if bn_size > 0:
            self.bn = nn.Linear(hidden_size, bn_size, bias=False)
            hidden_size = bn_size
        else:
            self.bn = None

        self.project = nn.Linear(hidden_size, output_size, bias=True)
        nn.init.xavier_normal_(self.emb.weight)
        if shared_emb and bn_size == 0:
            self.emb.weight = self.project.weight

    def forward(self, inputs):
        emb = self.emb(inputs) * self.scale
        emb = self.emb_drop(emb)

        if inputs.size(0) > 1:
            lengths = inputs.gt(0).sum(-1)
            emb = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
            out = self.lstm(emb)[0]
            out = pad_packed_sequence(out, batch_first=True)[0]
        else:
            out = self.lstm(emb)[0]

        out = self.emb_drop(out)
        out = self.bn(out) if self.bn is not None else out
        out = self.project(out)

        return out
        
    def decode(self, inputs):
        emb = self.emb(inputs) * self.scale
        out = self.lstm(emb)[0]
        out = self.bn(out) if self.bn is not None else out
        logit = self.project(out)
        logit = logit[:,-1,:].squeeze(1)

        return torch.log_softmax(logit, -1)
