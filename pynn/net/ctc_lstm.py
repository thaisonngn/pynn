# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math

import torch
import torch.nn as nn

from . import XavierLinear
from .s2s_lstm import Encoder

class DeepLSTM(nn.Module):
    def __init__(self, n_classes, d_input, d_model, n_layer, unidirect=False, d_project=0,
            dropout=0.1, dropconnect=0., time_ds=1, use_cnn=False, freq_kn=3, freq_std=2):
        super().__init__()

        self.encoder = Encoder(d_input, d_model, n_layer, unidirect=unidirect, time_ds=time_ds,
                            use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std,
                            dropout=dropout, dropconnect=dropconnect)
        d_project = d_model if d_project==0 else d_project
        self.project = None if d_project==d_model else XavierLinear(d_model, d_project)
        self.output = nn.Linear(d_project, n_classes, bias=True)

    def forward(self, seq, mask=None):
        out, mask = self.encoder(seq, mask)[0:2]
        out = self.project(out) if self.project is not None else out
        out = self.output(out)
        return out, mask

    def decode(self, x, mask=None):
        logit, mask = self.forward(x, mask)
        return torch.log_softmax(logit, -1), mask
