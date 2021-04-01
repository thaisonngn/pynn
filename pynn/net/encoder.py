# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import torch.nn as nn

from . import XavierLinear
from .s2s_transformer import Encoder as AttnEncoder
from .s2s_lstm import Encoder as LSTMEncoder

class SelfAttnNet(nn.Module):
    def __init__(self, n_vocab, d_input, d_model, d_inner=0, n_head=8, n_layer=4, d_project=0,
            rel_pos=False, unidirect=False, n_emb=0, use_cnn=False, time_ds=1, dropout=0.2, layer_drop=0.):
        super().__init__()
        d_inner = d_model*4 if d_inner==0 else d_inner
        embedding = n_emb > 0
        self.encoder = AttnEncoder(d_input, d_model, n_layer, n_head, d_inner, rel_pos=rel_pos,
                                   attn_mode=(1 if unidirect else 0), embedding=embedding, emb_vocab=n_emb, 
                                   use_cnn=use_cnn, time_ds=time_ds, dropout=dropout, layer_drop=layer_drop)
        d_project = d_model if d_project==0 else d_project
        self.project = None if d_project==d_model else XavierLinear(d_model, d_project)
        self.output = nn.Linear(d_project, n_vocab, bias=True)

    def forward(self, seq, mask=None):
        out, mask = self.encoder(seq, mask)[0:2]
        out = self.project(out) if self.project is not None else out
        out = self.output(out)
        return out, mask

    def decode(self, x, mask=None):
        logit, mask = self.forward(x, mask)
        return torch.log_softmax(logit, -1), mask

    def encode(self, seq, mask, hid=None):
        return self.encoder(seq, mask)

    def decode_ctc(self, enc_out, enc_mask):
        logit = self.project(enc_out) if self.project is not None else enc_out
        logit = self.output(logit)
        return torch.log_softmax(logit, -1)

    def align(self, s2s_model, seq, mask, tgt):
        tgt = tgt[:, 1:-1].contiguous()
        logit, mask = self.forward(seq, mask)
        return logit, mask, logit, tgt

class LSTMNet(nn.Module):
    def __init__(self, n_vocab, d_input, d_model, d_inner=0, n_head=8, n_layer=4, d_project=0,
            rel_pos=False, use_cnn=False, time_ds=1, dropout=0.2, dropconnect=0., layer_drop=0.):
        super().__init__()
        self.encoder = LSTMEncoder(d_input, d_model, n_layer, dropout=dropout,
                                   dropconnect=dropconnect, time_ds=time_ds, use_cnn=use_cnn)
        d_project = d_model if d_project==0 else d_project
        self.project = None if d_project==d_model else XavierLinear(d_model, d_project)
        self.output = nn.Linear(d_project, n_vocab, bias=True)

    def forward(self, seq, mask=None):
        out, mask = self.encoder(seq, mask)[0:2]
        out = self.project(out) if self.project is not None else out
        out = self.output(out)
        return out, mask

    def decode(self, x, mask=None):
        logit, mask = self.forward(x, mask)
        return torch.log_softmax(logit, -1), mask

    def encode(self, seq, mask, hid=None):
        return self.encoder(seq, mask)

    def decode_ctc(self, enc_out, enc_mask):
        logit = self.project(enc_out) if self.project is not None else enc_out
        logit = self.output(logit)
        return torch.log_softmax(logit, -1)
