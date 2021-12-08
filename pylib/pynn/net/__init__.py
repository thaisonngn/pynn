# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import torch.nn as nn

class XavierLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=False, dropout=0.):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias: self.linear.bias.data.zero_()
        self.drop = nn.Dropout(dropout)

    def share(self, linear):
        self.linear.weight = linear.weight

    def forward(self, x):
        x = self.linear(x)
        return self.drop(x)

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class WordDropout(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def forward(self, seq):
        dropout = self.dropout
        if dropout > 0. and self.training:
            mask = torch.empty((seq.size(0), seq.size(1), 1), device=seq.device, dtype=seq.dtype)
            seq = seq * mask.bernoulli_(1. - dropout) / (1. - dropout)
        return seq

def freeze_module(layer):
    for param in layer.parameters():
        param.requires_grad = False
    return layer
