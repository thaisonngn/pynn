# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import random

import torch
import torch.nn as nn

from . import XavierLinear, Swish
from .attn import PositionalEmbedding
from .transformer_layer import EncoderLayer, DecoderLayer

class Encoder(nn.Module):
    def __init__(self, d_input, d_model, n_layer, n_head, d_inner,
            attn_mode=0, rel_pos=False, dropout=0.1, layer_drop=0.,
            embedding=False, emb_vocab=0, emb_drop=0.,
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2):
        super().__init__()

        if embedding:
            self.emb = nn.Embedding(emb_vocab, d_input, padding_idx=0)
            self.emb_drop = nn.Dropout(emb_drop)
        else:
            self.emb = None

        self.time_ds = time_ds
        if use_cnn:
            cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)), Swish(),
                   nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)), Swish()]
            self.cnn = nn.Sequential(*cnn)
            d_input = ((((d_input - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1)*32
        else:
            self.cnn = None

        self.project = XavierLinear(d_input, d_model) if d_input!=d_model else None
        self.pos = PositionalEmbedding(d_model)
        self.scale = d_model**0.5
        self.rel_pos = rel_pos

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, dropout, rel_pos=rel_pos)
            for _ in range(n_layer)])

        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_drop = layer_drop
        self.attn_mode = attn_mode

    def get_mask(self, mode, src_mask):
        lt = src_mask.size(1)
        slf_mask = src_mask.eq(0)
        slf_mask = slf_mask.unsqueeze(1).expand(-1, lt, -1) # b x lq x lk

        if mode == 1:
            tri_mask = torch.ones((lt, lt), device=src_mask.device, dtype=torch.uint8)
            tri_mask = torch.triu(tri_mask, diagonal=1)
            tri_mask = tri_mask.unsqueeze(0).expand(src_mask.size(0), -1, -1)
            slf_mask = (slf_mask + tri_mask).gt(0)

        return slf_mask

    def forward(self, src_seq, src_mask):
        # -- Forward
        if self.emb is not None:
            src_seq = self.emb_drop(self.emb(src_seq))

        if self.time_ds > 1:
            x, ds = src_seq, self.time_ds
            l = ((x.size(1) - ds + 1) // ds) * ds
            x = x[:, :l, :]
            src_seq = x.view(x.size(0), -1, x.size(2)*ds)
            if src_mask is not None: src_mask = src_mask[:, 0:src_seq.size(1)*ds:ds]

        if self.cnn is not None:
            src_seq = src_seq.unsqueeze(1)
            src_seq = self.cnn(src_seq)
            src_seq = src_seq.permute(0, 2, 1, 3).contiguous()
            src_seq = src_seq.view(src_seq.size(0), src_seq.size(1), -1)
            if src_mask is not None: src_mask = src_mask[:, 0:src_seq.size(1)*4:4]

        enc_out = self.project(src_seq) if self.project is not None else src_seq

        if self.rel_pos:
            pos_emb = self.pos.embed(src_mask)
        else:
            pos_seq = torch.arange(0, enc_out.size(1), device=enc_out.device, dtype=enc_out.dtype)

            pos_emb = self.pos(pos_seq, enc_out.size(0))
            enc_out = enc_out*self.scale + pos_emb

        # -- Prepare masks
        slf_mask = self.get_mask(self.attn_mode, src_mask)

        nl = len(self.layer_stack)
        for l, enc_layer in enumerate(self.layer_stack):
            scale = 1.
            if self.training:
                drop_level = (l+1.) * self.layer_drop / nl
                if random.random() < drop_level: continue
                scale = 1. / (1.-drop_level)
            
            enc_out = (enc_out, pos_emb) if self.rel_pos else enc_out
            enc_out = enc_layer(
                enc_out, slf_mask=slf_mask, scale=scale)
            
        enc_out = self.layer_norm(enc_out)
        return enc_out, src_mask

class Decoder(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, n_head, d_inner,
            rel_pos=False, dropout=0.1, emb_drop=0., layer_drop=0., shared_emb=True):

        super().__init__()

        self.emb = nn.Embedding(n_vocab, d_model, padding_idx=0)
        self.pos = PositionalEmbedding(d_model)
        self.scale = d_model**0.5
        self.emb_drop = nn.Dropout(emb_drop)
        self.rel_pos = rel_pos
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, dropout, rel_pos=rel_pos)
            for _ in range(n_layer)])

        self.output = nn.Linear(d_model, n_vocab, bias=True)
        #nn.init.xavier_normal_(self.project.weight)
        if shared_emb: self.emb.weight = self.output.weight
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_drop = layer_drop
            
    def forward(self, tgt_seq, enc_out, enc_mask):
        # -- Forward
        dec_out = self.emb(tgt_seq) * self.scale
        if self.rel_pos:
            pos_emb = self.pos.embed(tgt_seq.gt(0))
        else:
            pos_seq = torch.arange(0, dec_out.size(1),
                                   device=dec_out.device, dtype=dec_out.dtype)
            pos_emb = self.pos(pos_seq, dec_out.size(0))
            dec_out = dec_out + pos_emb
        dec_out = self.emb_drop(dec_out)

        lt = tgt_seq.size(1)
        # -- Prepare masks
        slf_mask = tgt_seq.eq(0).unsqueeze(1).expand(-1, lt, -1)
        tri_mask = torch.ones((lt, lt), device=dec_out.device, dtype=torch.uint8)
        tri_mask = torch.triu(tri_mask, diagonal=1)
        tri_mask = tri_mask.unsqueeze(0).expand(tgt_seq.size(0), -1, -1)
        slf_mask = (slf_mask + tri_mask).gt(0)

        attn_mask = enc_mask.eq(0).unsqueeze(1).expand(-1, lt, -1)

        attn = None
        nl = len(self.layer_stack)
        for l, dec_layer in enumerate(self.layer_stack):
            scale = 1.
            if self.training:
                drop_level = (l+1.) * self.layer_drop / nl
                if random.random() < drop_level: continue
                scale = 1. / (1.-drop_level)

            dec_out = (dec_out, pos_emb) if self.rel_pos else dec_out
            dec_out, attn = dec_layer(
                dec_out, enc_out, slf_mask=slf_mask,
                dec_enc_mask=attn_mask, scale=scale)
                        
        dec_out = self.layer_norm(dec_out)
        dec_out = self.output(dec_out)
        
        return dec_out, attn

class Transformer(nn.Module):
    def __init__(
            self,
            n_vocab=1000, n_emb=0, d_input=40, d_model=512, d_inner=2048,
            n_enc=8, n_enc_head=8, n_dec=4, n_dec_head=8,
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2,
            dropout=0.1, emb_drop=0., enc_drop=0.0, dec_drop=0.0,
            shared_emb=False, rel_pos=False):

        super().__init__()

        self.encoder = Encoder(
            d_input=d_input, d_model=d_model, d_inner=d_inner,
            n_layer=n_enc, n_head=n_enc_head, rel_pos=rel_pos,
            embedding=(n_emb>0), emb_vocab=n_emb, emb_drop=emb_drop,
            time_ds=time_ds, use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std,
            dropout=dropout, layer_drop=enc_drop)

        self.decoder = Decoder(
            n_vocab, d_model=d_model, d_inner=d_inner, n_layer=n_dec,
            n_head=n_dec_head, shared_emb=shared_emb, rel_pos=False,
            dropout=dropout, emb_drop=emb_drop, layer_drop=dec_drop)

    def forward(self, src_seq, src_mask, tgt_seq, encoding=True):
        if encoding:
            enc_out, enc_mask = self.encoder(src_seq, src_mask)
        else:
            enc_out, enc_mask = src_seq, src_mask
        dec_out = self.decoder(tgt_seq, enc_out, enc_mask)[0]
        return dec_out, enc_out, enc_mask
        
    def encode(self, src_seq, src_mask):
        return self.encoder(src_seq, src_mask)

    def decode(self, enc_out, enc_mask, tgt_seq):
        dec_out, attn = self.decoder(tgt_seq, enc_out, enc_mask)
        dec_out = dec_out[:,-1,:].squeeze(1)
        return torch.log_softmax(dec_out, -1), attn

    def coverage(self, enc_out, enc_mask, tgt_seq, attn=None):
        if attn is None:
            attn = self.decoder(tgt_seq, enc_out, enc_mask)[1]
        attn = attn.mean(dim=1).sum(dim=1)
        cov = attn.gt(0.5).float().sum(dim=1)
        return cov
