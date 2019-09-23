''' Define the Layers '''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class XavierLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias: self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) # (n*b) x lq x dk
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, shared_kv=False, dropout=0.1, norm=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_vs = None if shared_kv else nn.Linear(d_model, n_head*d_k, bias=False)
                
        self.attention = ScaledDotProductAttention(np.power(d_k, 0.5), dropout)

        self.fc = nn.Linear(n_head * d_k, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = norm
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k=None, mask=None, scale=1.0):
        d_k, n_head = self.d_k, self.n_head

        if self.norm:
            residual = q
            q = self.layer_norm(q)
        if k is None: k = q

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()

        k_ = k                
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk

        if self.w_vs is not None:
            v = self.w_vs(k_).view(sz_b, len_k, n_head, d_k)
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lv x dv
        else:
            v = k

        if mask is not None:
            sz = sz_b*n_head // mask.size(0)
            mask = mask.repeat(sz, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if self.norm:
            output = output*scale + residual

        return output
