''' Define the Layers '''
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .attn import MultiHeadAttention

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=80):
        super().__init__()
        # create constant 'pe' matrix with values
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000**(2*(i+1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        _, seq_len, d_model = x.size()
        x = x * math.sqrt(d_model)
        return x + Variable(self.pe[:,:seq_len], requires_grad=False)

class RecurrentEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.rnn = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1,
                                bidirectional=False, bias=False, batch_first=True)
    def forward(self, x):
        #p = self.rnn(x)[0]
        #return x + p
        return self.rnn(x)[0]

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

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
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, shared_kv=True, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, shared_kv, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, slf_mask=None, scale=1.):
        enc_output = self.slf_attn(enc_input, mask=slf_mask, scale=scale)
        enc_output = self.pos_ffn(enc_output, scale)

        return enc_output

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, shared_kv=True, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, shared_kv, dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, shared_kv, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec_input, enc_output, slf_mask=None,
            dec_enc_mask=None, scale=1.):
        dec_output = self.slf_attn(dec_input, mask=slf_mask, scale=scale)
        dec_output = self.enc_attn(dec_output, enc_output, mask=dec_enc_mask, scale=scale)
        dec_output = self.pos_ffn(dec_output, scale)

        return dec_output
