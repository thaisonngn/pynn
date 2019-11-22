# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import MultiHeadAttention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1, norm=True):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid, bias=False) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in, bias=False) # position-wise
        self.norm = norm
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, scale=1.):
        if self.norm:
            residual = x
            x = self.layer_norm(x)
        x = F.relu(self.w_1(x))
        x = self.dropout_1(x)
        x = self.w_2(x)
        x = self.dropout_2(x)
        if self.norm:
            x = x*scale + residual
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, shared_kv=True, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, shared_kv, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, slf_mask=None, scale=1.):
        enc_out = self.slf_attn(enc_input, mask=slf_mask, scale=scale)[0]
        enc_out = self.pos_ffn(enc_out, scale)

        return enc_out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, shared_kv=True, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, shared_kv, dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, shared_kv, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec_input, enc_out, slf_mask=None,
            dec_enc_mask=None, scale=1.):
        dec_out = self.slf_attn(dec_input, mask=slf_mask, scale=scale)[0]
        dec_out, attn = self.enc_attn(dec_out, enc_out, mask=dec_enc_mask, scale=scale)
        dec_out = self.pos_ffn(dec_out, scale)

        return dec_out, attn
