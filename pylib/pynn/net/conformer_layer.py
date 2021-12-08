# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import XavierLinear, Swish
from .attn import MultiHeadedAttention, RelPositionAttention

class ConformerFFN(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1, residual=False):
        super().__init__()

        self.w_1 = XavierLinear(d_in, d_hid) # position-wise
        self.w_2 = XavierLinear(d_hid, d_in) # position-wise
        self.activation = Swish()
        self.layer_norm = nn.LayerNorm(d_in)
        self.drop = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, scale=1.):
        residual = x if self.residual else None
        x = self.layer_norm(x)
        x = self.activation(self.w_1(x))
        x = self.drop(x)
        x = self.w_2(x)
        x = self.drop(x) * scale
        x = x if residual is None else (x + residual)
        return x

class ConformerConv(nn.Module):
    def __init__(self, channels, kernels, dropout=0.1, residual=False):
        super().__init__()

        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernels - 1) % 2 == 0

        self.layer_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(channels, 2*channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernels, stride=1,
                                        padding=(kernels - 1)//2, groups=channels, bias=False)

        self.batch_norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.activation = Swish()
        self.drop = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, scale=1.):
        """Compute convolution module.
        :param torch.Tensor x: (batch, time, size)
        :return torch.Tensor: convoluted `value` (batch, time, d_model)
        """
        # exchange the temporal dimension and the feature dimension
        residual = x if self.residual else None
        x = self.layer_norm(x).transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.batch_norm(x))

        x = self.pointwise_conv2(x).transpose(1, 2)
        x = self.drop(x) * scale
        x = x if residual is None else (x + residual)        
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, n_kernel=25, dropout=0.1, rel_pos=True):
        super(EncoderLayer, self).__init__()

        Attention = RelPositionAttention if rel_pos else MultiHeadedAttention
        self.slf_attn = Attention(n_head, d_model, dropout, residual=True)
        self.ffn_pre = ConformerFFN(d_model, d_inner, dropout, residual=True)
        self.ffn_pos = ConformerFFN(d_model, d_inner, dropout, residual=True)
        self.conv = ConformerConv(d_model, n_kernel, dropout, residual=True)

    def forward(self, x, pos, mask, drop_level):
        if self.training:
            scale = 1. / (1.-drop_level)
            if random.random() > drop_level:
                x = self.ffn_pre(x, scale)
            if random.random() > drop_level:
                v = x if pos is None else (x, pos)      
                x = self.slf_attn(v, mask=mask, scale=scale)[0]
            if random.random() > drop_level:
                x = self.conv(x, scale)
            if random.random() > drop_level:
                x = self.ffn_pos(x, scale)
        else:
            x = self.ffn_pre(x)
            v = x if pos is None else (x, pos)      
            x = self.slf_attn(v, mask=mask)[0]
            x = self.conv(x)
            x = self.ffn_pos(x)
        return x
