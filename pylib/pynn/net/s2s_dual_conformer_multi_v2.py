# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import XavierLinear, Swish
from .attn import PositionalEmbedding, MultiHeadedAttention
from .conformer_layer import EncoderLayer
from .transformer_layer import DecoderLayer

class Encoder(nn.Module):
    def __init__(self, d_input, d_model, d_inner, n_layer, n_head, n_kernel=25,
            dropout=0.1, layer_drop=0., time_ds=1, use_cnn=False, freq_kn=3, freq_std=2):
        super().__init__()

        self.time_ds = time_ds
        if use_cnn:
            cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)), Swish(),
                   nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)), Swish()]
            self.cnn = nn.Sequential(*cnn)
            d_input = ((((d_input - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1)*32
        else:
            self.cnn = None

        self.emb = XavierLinear(d_input, d_model)
        self.pos = PositionalEmbedding(d_model)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, n_kernel, dropout)
            for _ in range(n_layer)])

        self.norm = nn.LayerNorm(d_model)
        self.layer_drop = layer_drop

    def forward(self, src_seq, src_mask, hid=None):
        # -- Forward
        if self.cnn is not None:
            src_seq = src_seq.unsqueeze(1)
            src_seq = self.cnn(src_seq)
            src_seq = src_seq.permute(0, 2, 1, 3).contiguous()
            src_seq = src_seq.view(src_seq.size(0), src_seq.size(1), -1)
            if src_mask is not None: src_mask = src_mask[:, 0:src_seq.size(1)*4:4]

        enc_out = src_seq if self.emb is None else self.emb(src_seq)
        pos_emb = self.pos.embed(src_mask)
 
        # -- Prepare masks
        slf_mask = src_mask.eq(0)
        slf_mask = slf_mask.unsqueeze(1).expand(-1, src_mask.size(1), -1) # b x lq x lk

        nl = len(self.layer_stack)
        for l, enc_layer in enumerate(self.layer_stack):
            drop_level = (l+1.) * self.layer_drop / nl
            enc_out = enc_layer(enc_out, pos_emb, slf_mask, drop_level)
        enc_out = self.norm(enc_out)

        return enc_out, src_mask, hid

class Decoder(nn.Module):
    def __init__(self, n_classes, d_model, d_inner, n_layer, d_in, n_head=8,
            shared_emb=True, dropout=0.2, emb_drop=0., layer_drop=0.):
        super().__init__()

        self.emb = nn.Embedding(n_classes, d_model, padding_idx=0)
        self.pos = PositionalEmbedding(d_model)
        self.scale = d_model**0.5
        self.drop = nn.Dropout(emb_drop)
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, dropout, rel_pos=False)
            for _ in range(n_layer)])

        self.output = nn.Linear(d_model, n_classes, bias=False)
        if shared_emb: self.emb.weight = self.output.weight
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_drop = layer_drop
            
    def forward(self, tgt_seq, enc_out, enc_mask):
        lt = tgt_seq.size(1)
        emb_out = self.emb(tgt_seq)
        dec_out = emb_out * self.scale
        pos_seq = torch.arange(0, lt, device=enc_out.device, dtype=enc_out.dtype)
        pos_emb = self.pos(pos_seq, dec_out.size(0))
        dec_out = self.drop(dec_out + pos_emb)

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
            
            dec_out, attn = dec_layer(
                dec_out, enc_out, slf_mask=slf_mask,
                dec_enc_mask=attn_mask, scale=scale)
        
        dec_out = self.output(self.layer_norm(dec_out))
        
        return dec_out, emb_out

class Trancoder(nn.Module):
    def __init__(self, n_classes, d_model, d_inner, n_layer, d_in, n_head=8,
            dropout=0.2, emb_drop=0.):
        super().__init__()

        self.emb = nn.Embedding(n_classes, d_model, padding_idx=0)
        self.pos = PositionalEmbedding(d_model)
        self.scale = d_model**0.5
        self.drop = nn.Dropout(emb_drop)
        
        self.layer_stack_e = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, dropout, rel_pos=False)
            for _ in range(n_layer)])
        self.layer_norm_e = nn.LayerNorm(d_model)
        
        self.layer_stack_d = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, dropout, rel_pos=False)
            for _ in range(n_layer)])
        self.layer_norm_d = nn.LayerNorm(d_model)
        
        self.output = nn.Linear(d_model*2, n_classes, bias=False)
            
    def forward(self, tgt_seq, e_out, e_mask, d_out, d_mask):
        lt = tgt_seq.size(1)
        dec_out = self.emb(tgt_seq) * self.scale
        pos_seq = torch.arange(0, lt, device=dec_out.device, dtype=dec_out.dtype)
        pos_emb = self.pos(pos_seq, dec_out.size(0))
        dec_out = self.drop(dec_out + pos_emb)

        # -- Prepare masks
        slf_mask = tgt_seq.eq(0).unsqueeze(1).expand(-1, lt, -1)
        tri_mask = torch.ones((lt, lt), device=tgt_seq.device, dtype=torch.uint8)
        tri_mask = torch.triu(tri_mask, diagonal=1)
        tri_mask = tri_mask.unsqueeze(0).expand(tgt_seq.size(0), -1, -1)
        slf_mask = (slf_mask + tri_mask).gt(0)
        
        attn_mask = e_mask.eq(0).unsqueeze(1).expand(-1, lt, -1)
        layer_e = dec_out
        for dec_layer in self.layer_stack_e:
            layer_e, attn = dec_layer(
                layer_e, e_out, slf_mask=slf_mask,
                dec_enc_mask=attn_mask)
        layer_e = self.layer_norm_e(layer_e)
        
        attn_mask = d_mask.eq(0).unsqueeze(1).expand(-1, lt, -1)
        layer_d = dec_out
        for dec_layer in self.layer_stack_d:
            layer_d, attn = dec_layer(
                layer_d, d_out, slf_mask=slf_mask,
                dec_enc_mask=attn_mask)
        layer_d = self.layer_norm_d(layer_d)

        emb_out = torch.cat((layer_e, layer_d), dim=-1)
        dec_out = self.output(emb_out)
        
        return dec_out

class Conformer(nn.Module):
    def __init__(self, d_input, n_dec_vocab, n_tran_vocab, d_enc=256, d_inner=0, n_enc=4,
                 n_kernel=25, d_dec=320, n_dec=2, n_head=8, shared_emb=True,
                 dropout=0.1, emb_drop=0.1, enc_drop=0., dec_drop=0.,
                 time_ds=1, use_cnn=False, freq_kn=3, freq_std=2):
        super().__init__()

        d_inner = d_enc*4 if d_inner == 0 else d_inner
        self.encoder = Encoder(d_input, d_enc, d_inner, n_enc, n_head, n_kernel=n_kernel,
                            dropout=dropout, layer_drop=enc_drop,
                            time_ds=time_ds, use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std)
        self.decoder = Decoder(n_dec_vocab, d_dec, d_inner, n_dec, d_enc, n_head, shared_emb=shared_emb,
                            dropout=dropout, emb_drop=emb_drop, layer_drop=dec_drop)
        self.trancoder = Trancoder(n_tran_vocab, d_dec, d_inner, n_dec, d_enc, n_head,
                            dropout=dropout, emb_drop=emb_drop)

    def attend(self, src_seq, src_mask, tgt_seq):
        enc_out, src_mask = self.encoder(src_seq, src_mask)[0:2]
        logit, attn = self.decoder(tgt_seq, enc_out, src_mask)[0:2]
        return logit, attn, src_mask

    def forward(self, src_seq, src_mask, tgt_pre, tgt_pos, encoding=True):
        if encoding:
            enc_out, enc_mask = self.encoder(src_seq, src_mask)[0:2]
        else:
            enc_out, enc_mask = src_seq, src_mask

        dec_out, emb_out = self.decoder(tgt_pre, enc_out, enc_mask)[0:2]
        tran_out = self.trancoder(tgt_pos, enc_out, enc_mask, emb_out, tgt_pre.gt(0))
        return dec_out, tran_out, enc_out, src_mask

    def encode(self, src_seq, src_mask, hid=None):
        return self.encoder(src_seq, src_mask, hid)

    def decode(self, enc_out, enc_mask, tgt_seq):
        logit, emb_out = self.decoder(tgt_seq, enc_out, enc_mask)[0:2]
        logit = logit[:,-1,:].squeeze(1)
        return torch.log_softmax(logit, -1), emb_out

    def trancode(self, enc_out, enc_mask, emb_out, emb_mask, tgt_seq):
        logit = self.trancoder(tgt_seq, enc_out, enc_mask, emb_out, emb_mask)
        return torch.log_softmax(logit, -1)
