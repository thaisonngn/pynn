# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import XavierLinear, Swish

from .attn import MultiHeadedAttention, RelPositionAttention

class PositionwiseFF(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1, residual=False):
        super().__init__()
        self.w_1 = XavierLinear(d_in, d_hid) # position-wise
        self.w_2 = XavierLinear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.activation = Swish()
        self.drop = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, scale=1.):
        residual = x if self.residual else None
        x = self.layer_norm(x)
        x = self.activation(self.w_1(x))
        x = self.drop(x)
        x = self.w_2(x)
        x = self.drop(x)*scale
        x = x if residual is None else (x + residual)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout=0.1, rel_pos=False):
        super().__init__()
        Attention = RelPositionAttention if rel_pos else MultiHeadedAttention
        self.slf_attn = Attention(n_head, d_model, dropout, residual=True)
        self.pos_ffn = PositionwiseFF(d_model, d_inner, dropout, residual=True)

    def forward(self, enc_inp, slf_mask=None, scale=1.):
        enc_out = self.slf_attn(enc_inp, mask=slf_mask, scale=scale)[0]
        enc_out = self.pos_ffn(enc_out, scale=scale)

        return enc_out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout=0.1, n_enc_head=0, rel_pos=False):
        super().__init__()
        Attention = RelPositionAttention if rel_pos else MultiHeadedAttention
        self.slf_attn = Attention(n_head, d_model, dropout, residual=True)
        n_head = n_enc_head if n_enc_head > 0 else n_head
        self.enc_attn = MultiHeadedAttention(n_head, d_model, dropout, residual=True)
        self.pos_ffn = PositionwiseFF(d_model, d_inner, dropout, residual=True)

    def forward(self, dec_inp, enc_out, slf_mask=None, dec_enc_mask=None, scale=1.):
        dec_out = self.slf_attn(dec_inp, mask=slf_mask, scale=scale)[0]
        dec_out, attn = self.enc_attn(dec_out, enc_out, mask=dec_enc_mask, scale=scale)
        dec_out = self.pos_ffn(dec_out, scale=scale)

        return dec_out, attn
