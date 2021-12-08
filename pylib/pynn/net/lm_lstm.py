# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import XavierLinear
from .rnn import LSTM

class SeqLM(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, d_emb=0, d_project=0, shared_emb=True,
            dropout=0.2, dropconnect=0., emb_drop=0., layer_drop=0.):
        super().__init__()

        # Define layers
        d_emb = d_model if d_emb==0 else d_emb
        self.emb = nn.Embedding(n_vocab, d_emb, padding_idx=0)
        self.scale = d_emb**0.5
        self.emb_drop = nn.Dropout(emb_drop)
        self.dropconnect = dropconnect
 
        #self.lstm = nn.LSTM(d_emb, d_model, n_layer, dropout=dropout, batch_first=True)
        self.lstm = LSTM(d_emb, d_model, n_layer, batch_first=True,
                         dropout=dropout, dropconnect=dropconnect)

        d_project = d_model if d_project==0 else d_project
        self.project = None if d_project==d_model else XavierLinear(d_model, d_project)
        self.output = nn.Linear(d_project, n_vocab, bias=False)
        #nn.init.xavier_normal_(self.emb.weight)
        if shared_emb: self.emb.weight = self.output.weight

    def forward(self, inputs):
        emb = self.emb_drop(self.emb(inputs) * self.scale)

        if inputs.size(0) > 1:
            lengths = inputs.gt(0).sum(-1)
            emb = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out = self.lstm(emb)[0]
            out = pad_packed_sequence(out, batch_first=True)[0]
        else:
            out = self.lstm(emb)[0]

        out = self.project(out) if self.project is not None else out
        out = self.output(out)

        return out
        
    def decode(self, inputs):
        logits = self.forward(inputs)
        logits = logits[:,-1,:].squeeze(1)
        return torch.log_softmax(logits, -1)
