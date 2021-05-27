# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import XavierLinear, Swish
from .rnn import LSTM
from .attn import LocationAwareAttention
from .s2s_lstm import Encoder

class Decoder(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, d_emb=0, d_project=0,
            shared_emb=True, dropout=0.2, dropconnect=0., emb_drop=0., pack=True):
        super().__init__()

        # Define layers
        d_emb = d_model if d_emb==0 else d_emb
        self.emb = nn.Embedding(n_vocab, d_emb, padding_idx=0)
        self.emb_drop = nn.Dropout(emb_drop)
        self.scale = d_emb**0.5

        self.attn = LocationAwareAttention(d_model, 256)
        dropout = (0 if n_layer == 1 else dropout)
        self.lstm = LSTM(d_emb, d_model, n_layer, batch_first=True, dropout=dropout, dropconnect=dropconnect)
        d_project = d_model if d_project==0 else d_project
        self.project = None if d_project==d_model else XavierLinear(d_model, d_project)
        self.output = nn.Linear(d_project, n_vocab, bias=False)
        if shared_emb: self.emb.weight = self.output.weight
        self.pack = pack

    def forward(self, dec_in, hid, ctx, attn, enc_out, enc_mask):
        dec_emb = self.emb(dec_in) * self.scale
        dec_emb = self.emb_drop(dec_emb + ctx)
        dec_emb, hid = self.lstm(dec_emb, hid)
        ctx, attn = self.attn(dec_emb, attn, enc_out, enc_mask)
        dec_out = (ctx + dec_emb)
        dec_out = self.project(dec_out) if self.project is not None else dec_out
        dec_out = self.output(dec_out)

        return dec_out, hid, ctx, attn

        
class Seq2Seq(nn.Module):
    def __init__(self, n_vocab=1000, d_input=40, d_enc=320, n_enc=6, d_dec=320, n_dec=2,
            unidirect=False, incl_win=0, d_emb=0, d_project=0, n_head=8, shared_emb=False, 
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2, enc_dropout=0.2, enc_dropconnect=0., 
            dec_dropout=0.1, dec_dropconnect=0., emb_drop=0., pack=True):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(d_input, d_enc, n_enc,
                            unidirect=unidirect, incl_win=incl_win, time_ds=time_ds,
                            use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std,
                            dropout=enc_dropout, dropconnect=enc_dropconnect, pack=pack)
        self.decoder = Decoder(n_vocab, d_dec, n_dec,
                            d_emb=d_emb, d_project=d_project, shared_emb=shared_emb,
                            dropout=dec_dropout, dropconnect=dec_dropconnect, emb_drop=emb_drop, pack=pack)
        self.transform = None if d_dec==d_enc else XavierLinear(d_enc, d_dec)

    def attend(self, src_seq, src_mask, tgt_seq):
        enc_out, enc_mask = self.encoder(src_seq, src_mask)[0:2]
        logit, attn = self.decoder(tgt_seq, enc_out, enc_mask)[0:2]
        return logit, attn, enc_mask

    def forward(self, src_seq, src_mask, tgt_seq, encoding=True):
        if encoding:
            enc_out, enc_mask = self.encoder(src_seq, src_mask)[0:2]
        else:
            enc_out, enc_mask = src_seq, src_mask
        enc = self.transform(enc_out) if self.transform is not None else enc_out
        mask = enc_mask.eq(0)
        bs, ds, es = tgt_seq.size(0), tgt_seq.size(1), enc.size(1)
        
        dec_out, hid = [], None
        ctx = torch.zeros((bs, 1, enc.size(-1)), dtype=enc.dtype, device=enc.device)
        attn = torch.zeros((bs, es), dtype=enc.dtype, device=enc.device)
        
        for t in range(0, ds):
            d_in = tgt_seq[:,t].unsqueeze(1)
            d_out, hid, ctx, attn = self.decoder(d_in, hid, ctx, attn, enc, mask)
            dec_out.append(d_out)
        dec_out = torch.cat(dec_out, dim=1)
        
        return dec_out, enc_out, enc_mask

    def encode(self, src_seq, src_mask, hid=None):
        enc_out, enc_mask, hid = self.encoder(src_seq, src_mask, hid)
        enc = self.transform(enc_out) if self.transform is not None else enc_out
        mask = enc_mask.eq(0)
        return enc, mask, hid

    def decode(self, enc, mask, tgt_seq, state=None):
        bs, es = tgt_seq.size(0), enc.size(1)
        if state is not None:
            hid, cell, ctx, attn = zip(*state)
            d_hid = (torch.stack(hid, dim=1), torch.stack(cell, dim=1))
            ctx, attn = torch.stack(ctx, dim=0), torch.stack(attn, dim=0)
        else:
            d_hid = None
            ctx = torch.zeros((bs, 1, enc.size(-1)), dtype=enc.dtype, device=enc.device)
            attn = torch.zeros((bs, es), dtype=enc.dtype, device=enc.device)
        logit, d_hid, ctx, attn = self.decoder(tgt_seq, d_hid, ctx, attn, enc, mask)
        hid, cell = d_hid
        state = [(hid[:,j,:], cell[:,j,:], ctx[j,:,:], attn[j,:]) for j in range(bs)]
        logit = logit[:,-1,:].squeeze(1)
        return torch.log_softmax(logit, -1), attn.unsqueeze(0), state

    def coverage(self, enc_out, enc_mask, tgt_seq, attn=None):
        if attn is None:
            attn = self.decoder(tgt_seq, enc_out, enc_mask)[1]
        attn = attn.mean(dim=1).sum(dim=1)
        cov = attn.gt(0.5).float().sum(dim=1)
        return cov

    def get_attn(self, enc_out, enc_mask, tgt_seq):
        return self.decoder(tgt_seq, enc_out, enc_mask)[1]
        
    def get_logit(self, enc_out, enc_mask, tgt_seq):
        logit = self.decoder(tgt_seq, enc_out, enc_mask)[0]
        return torch.log_softmax(logit, -1)
