# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import random
import numpy as np

import torch
import torch.nn as nn

from .attn import MultiHeadAttention, PositionalEmbedding
from .tf_layer import EncoderLayer, DecoderLayer

def get_attn_pad(pad_mask):
    ''' For masking out the padding part of key sequence. '''
    len_q = pad_mask.size(1)
    padding_mask = pad_mask.eq(0)
    return padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    
def get_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    return padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

def get_sequent_mask(seq, flip=False):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, *_ = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    if flip: subsequent_mask = subsequent_mask.flip(0)
    return subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

class Encoder(nn.Module):
    def __init__(
            self,
            d_input, n_layers, n_head, d_k, d_model, d_inner,
            dropout=0.1, layer_drop=0., shared_kv=False, attn_mode=0,
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2):

        super().__init__()

        self.time_ds = time_ds
        if use_cnn:
            cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)),
                   nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std))]
            self.cnn = nn.Sequential(*cnn)
            d_input = ((((d_input - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1)*32
        else:
            self.cnn = None

        self.emb = nn.Linear(d_input, d_model, bias=False)
        #nn.init.xavier_normal_(self.emb.weight)

        self.pe = PositionalEmbedding(d_model, 3000)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, shared_kv, dropout)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_drop = layer_drop
        self.attn_mode = attn_mode

    def get_attn_mask(self, src_seq, src_mask):
        slf_mask = get_attn_pad(src_mask)
        if self.attn_mode == 1:
            sequent_mask = get_sequent_mask(src_seq)
            slf_mask = (slf_mask + sequent_mask).gt(0)
        elif self.attn_mode == 2:
            sequent_mask = get_sequent_mask(src_seq, flip=True)
            fwd_mask = (slf_mask + sequent_mask).gt(0)
            sequent_mask = get_sequent_mask(src_seq, flip=False)
            bwd_mask = (slf_mask + sequent_mask).gt(0)
            slf_mask = torch.cat([bwd_mask, bwd_mask])
        return slf_mask

    def forward(self, src_seq, src_mask):
        # -- Forward
        if self.time_ds > 1:
            x, ds = src_seq, self.time_ds
            l = ((x.size(1) - 3) // ds) * ds
            x = x[:, :l, :]
            src_seq = x.view(x.size(0), -1, x.size(2)*ds)
            if src_mask is not None: src_mask = src_mask[:, 0:src_seq.size(1)*ds:ds]
                
        if self.cnn is not None:
            src_seq = src_seq.unsqueeze(1)
            src_seq = self.cnn(src_seq)
            src_seq = src_seq.permute(0, 2, 1, 3).contiguous()
            src_seq = src_seq.view(src_seq.size(0), src_seq.size(1), -1)
            if src_mask is not None: src_mask = src_mask[:, 0:src_seq.size(1)*4:4]

        enc_out = self.pe(self.emb(src_seq))

        # -- Prepare masks
        slf_mask = self.get_attn_mask(src_seq, src_mask)

        nl = len(self.layer_stack)
        for l, enc_layer in enumerate(self.layer_stack):
            scale = 1.
            if self.training:
                drop_level = (l+1.) * self.layer_drop / nl
                if random.random() < drop_level: continue
                scale = 1. / (1.-drop_level)
                
            enc_out = enc_layer(
                enc_out, slf_mask=slf_mask, scale=scale)
            
        enc_out = self.layer_norm(enc_out)
        return enc_out, src_mask

class Decoder(nn.Module):
    def __init__(
            self,
            n_vocab, n_layers, n_head, d_k, d_model, d_inner,
            dropout=0.1, emb_drop=0., layer_drop=0., shared_kv=False, shared_emb=True):

        super().__init__()

        self.emb = nn.Embedding(
            n_vocab, d_model, padding_idx=0)
        self.pe = PositionalEmbedding(d_model, 500)
        self.emb_drop = nn.Dropout(emb_drop)
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, shared_kv, dropout)
            for _ in range(n_layers)])

        self.project = nn.Linear(d_model, n_vocab, bias=True)
        #nn.init.xavier_normal_(self.project.weight)
        if shared_emb: self.emb.weight = self.project.weight
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_drop = layer_drop
            
    def forward(self, tgt_seq, enc_out, src_mask):
        # -- Prepare masks
        slf_mask_subseq = get_sequent_mask(tgt_seq)
        slf_mask_keypad = get_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_mask = (slf_mask_keypad + slf_mask_subseq).gt(0)

        dec_enc_mask = get_key_pad_mask(seq_k=src_mask, seq_q=tgt_seq)

        # -- Forward
        dec_out = self.pe(self.emb(tgt_seq))
        dec_out = self.emb_drop(dec_out)

        nl = len(self.layer_stack)
        for l, dec_layer in enumerate(self.layer_stack):
            scale = 1.
            if self.training:
                drop_level = (l+1.) * self.layer_drop / nl
                if random.random() < drop_level: continue
                scale = 1. / (1.-drop_level)
                
            dec_out, attn = dec_layer(
                dec_out, enc_out, slf_mask=slf_mask,
                dec_enc_mask=dec_enc_mask, scale=scale)
                        
        dec_out = self.layer_norm(dec_out)
        dec_out = self.project(dec_out)
        
        return dec_out, attn

class Transformer(nn.Module):
    def __init__(
            self,
            n_vocab=1000, d_input=40, d_model=512, d_inner=2048,
            n_enc=8, n_enc_head=8, n_dec=4, n_dec_head=8,
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2,
            dropout=0.1, emb_drop=0., enc_drop=0.0, dec_drop=0.0,
            shared_kv=False, shared_emb=False, attn_mode=0):

        super().__init__()

        self.encoder = Encoder(
            d_input=d_input, d_model=d_model, d_inner=d_inner,
            n_layers=n_enc, n_head=n_enc_head, d_k=d_model//n_enc_head,
            time_ds=time_ds,use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std,
            dropout=dropout, layer_drop=enc_drop,
            shared_kv=shared_kv, attn_mode=attn_mode)

        self.decoder = Decoder(
            n_vocab, d_model=d_model, d_inner=d_inner,
            n_layers=n_dec, n_head=n_dec_head, d_k=d_model//n_dec_head,
            dropout=dropout, emb_drop=emb_drop, layer_drop=dec_drop,
            shared_emb=shared_emb)

    def forward(self, src_seq, src_mask, tgt_seq):
        enc_out, src_mask = self.encoder(src_seq, src_mask)
        dec_out = self.decoder(tgt_seq, enc_out, src_mask)[0]
        
        return dec_out.view(-1, dec_out.size(2))
        
    def encode(self, src_seq, src_mask):
        return self.encoder(src_seq, src_mask)

    def decode(self, enc_out, src_mask, tgt_seq):
        dec_out = self.decoder(tgt_seq, enc_out, src_mask)[0]
        dec_out = dec_out[:,-1,:].squeeze(1)
        return torch.log_softmax(dec_out, -1)
        
    def converage(self, enc_out, src_mask, tgt_seq, alpha=0.05):
        attn =  dec_out = self.decoder(tgt_seq, enc_out, src_mask)[1]
        cs = torch.cumsum(attn, dim=-1)
        cs = cs.le(1.-alpha) - cs.le(alpha)
        lens = cs.sum(dim=-1).argmin(dim=1).unsqueeze(1)
        lens = lens.expand(-1, attn.size(1), -1)
        ids = torch.arange(attn.size(1), device=enc_out.device)
        ids = ids.unsqueeze(0).unsqueeze(-1)
        ids = ids.expand(attn.size(0), -1, attn.size(2))
        ids = ids.eq(lens).unsqueeze(-1).expand(-1, -1, -1, attn.size(3))
        cov = torch.masked_select(cs, ids)
        cov = cov.view(cs.size(0), cs.size(2), cs.size(3))
        cov = cov.sum(dim=1).gt(0).sum(dim=1).float()

        return cov        

