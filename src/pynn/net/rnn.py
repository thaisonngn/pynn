# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

def _weight_drop(module, weights, dropout):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    original_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = F.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)
        out = original_forward(*args, **kwargs)
        for name_w in weights:
            delattr(module, name_w)
        return out

    setattr(module, 'forward', forward)

class LSTM(torch.nn.LSTM):
    def __init__(self, *args, dropconnect=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, dropconnect)

    def flatten_parameters(*args, **kwargs):
        # Learn from https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
        # Replace flatten_parameters with nothing
        return
