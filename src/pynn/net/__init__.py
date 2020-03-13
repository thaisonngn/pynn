# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import torch.nn as nn

class WordDropout(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def forward(self, seq):
        dropout = self.dropout
        if dropout > 0. and self.training:
            mask = torch.Tensor(seq.size(0), seq.size(1), 1).to(seq.device)
            mask = mask.type(seq.dtype).bernoulli_(1. - dropout) / (1. - dropout)
            seq *= mask
        return seq
