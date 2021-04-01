# Copyright 2020 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from . import XavierLinear, Swish

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEmbedding, self).__init__()

        inv_freq = 1. / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def embed(self, mask):
        seq_len = mask.sum(-1)
        seq_len = seq_len.max() - seq_len
        idx = torch.arange(mask.size(1)-1, -1, -1, device=mask.device)
        idx = idx.unsqueeze(0) - seq_len.unsqueeze(1)
        idx = idx * idx.gt(0)

        idx = idx.view(-1).type(self.inv_freq.dtype)
        sinusoid_inp = torch.ger(idx, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = pos_emb.view(*mask.size(), pos_emb.size(-1))
        return Variable(pos_emb, requires_grad=False)

    def forward(self, pos_seq, bsz=None):
        inv_freq = self.inv_freq.type(pos_seq.dtype)
        sinusoid_inp = torch.ger(pos_seq, inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            pos_emb = pos_emb.unsqueeze(0).expand(bsz, -1, -1)
        return Variable(pos_emb, requires_grad=False)

class SinusoidPosition(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # create constant 'pe' matrix with values
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000**(i/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000**((i+1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, seq):
        bsz, seq_len = seq.size(0), seq.size(1)
        pos_emb = self.pe[:, :seq_len, :].expand(bsz, -1, -1)
        return Variable(pos_emb, requires_grad=False)

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout=0.1, residual=False):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.norm = nn.LayerNorm(n_feat)
        self.w_q = XavierLinear(n_feat, n_feat)
        self.w_k = XavierLinear(n_feat, n_feat)
        self.w_v = XavierLinear(n_feat, n_feat)
        self.w_out = XavierLinear(n_feat, n_feat)
        self.drop = nn.Dropout(dropout)
        self.residual = residual

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.w_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.w_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.w_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, scores.size(1), -1, -1)            
            scores = scores.masked_fill(mask, -np.inf)
        attn =  torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        attn = self.drop(attn)
        x = torch.matmul(attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous()
        x = x.view(n_batch, -1, self.h*self.d_k) # (batch, time1, d_model)
        x = self.w_out(x) # (batch, time1, d_model)

        return x, attn

    def forward(self, query, key=None, mask=None, scale=1.):
        residual = query if self.residual else None
        query = self.norm(query)
        if key is None: key = query
        value = key

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        x, attn = self.forward_attention(v, scores, mask)
        x = self.drop(x) * scale
        x = x if residual is None else (x + residual)
        return x, attn


class RelPositionAttention(MultiHeadedAttention):
    def __init__(self, n_head, n_feat, dropout=0.1, residual=False):
        super().__init__(n_head, n_feat, dropout, residual)
        # linear transformation for positional ecoding
        self.w_pos = XavierLinear(n_feat, n_feat)
        self.p_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.p_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        nn.init.xavier_uniform_(self.p_u)
        nn.init.xavier_uniform_(self.p_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        return x

    def forward(self, query_pos, key=None, mask=None, scale=1.):
        query, pos_emb = query_pos
        residual = query if self.residual else None
        query = self.norm(query) 
        if key is None: key = query
        value = key

        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.w_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_u = (q + self.p_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_v = (q + self.p_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        x, attn = self.forward_attention(v, scores, mask)
        x = self.drop(x) * scale
        x = x if residual is None else (x + residual)
        return x, attn

