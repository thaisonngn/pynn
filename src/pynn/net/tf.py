''' Define the Transformer model '''
import math
import random
import numpy as np

import torch
import torch.nn as nn

from .attn import MultiHeadAttention
from .tf_layer import PositionalEmbedding, EncoderLayer, DecoderLayer

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
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            d_input, n_layers, n_head, d_k, d_model, d_inner,
            dropout=0.1, layer_drop=0., shared_kv=False, attn_mode=0, use_cnn=False, freq_kn=3, freq_std=2):

        super().__init__()

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
            sequent_mask = get_sequent_mask(src_seq, True)
            fwd_mask = (slf_mask + sequent_mask).gt(0)
            sequent_mask = get_sequent_mask(src_seq, False)
            bwd_mask = (slf_mask + sequent_mask).gt(0)
            slf_mask = torch.cat([bwd_mask, bwd_mask])
        return slf_mask

    def forward(self, src_seq, src_mask):
        # -- Forward
        if self.cnn is not None:
            src_seq = src_seq.unsqueeze(1)
            src_seq = self.cnn(src_seq)
            src_seq = src_seq.permute(0, 2, 1, 3).contiguous()
            src_seq = src_seq.view(src_seq.size(0), src_seq.size(1), -1)
            if src_mask is not None: src_mask = src_mask[:, 0:src_seq.size(1)*4:4]

        enc_output = self.pe(self.emb(src_seq))

        # -- Prepare masks
        slf_mask = self.get_attn_mask(src_seq, src_mask)

        nl = len(self.layer_stack)
        for l, enc_layer in enumerate(self.layer_stack):
            scale = 1.
            if self.training:
                drop_level = (l+1.) * self.layer_drop / nl
                if random.random() < drop_level: continue
                scale = 1. / (1.-drop_level)
                
            enc_output = enc_layer(
                enc_output, slf_mask=slf_mask, scale=scale)
            
        enc_output = self.layer_norm(enc_output)
        return enc_output, src_mask

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

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
        if shared_emb: self.emb.weight = self.project.weight
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_drop = layer_drop
            
    def forward(self, tgt_seq, enc_output, src_mask):
        # -- Prepare masks
        slf_mask_subseq = get_sequent_mask(tgt_seq)
        slf_mask_keypad = get_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_mask = (slf_mask_keypad + slf_mask_subseq).gt(0)

        dec_enc_mask = get_key_pad_mask(seq_k=src_mask, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.pe(self.emb(tgt_seq))
        dec_output = self.emb_drop(dec_output)

        nl = len(self.layer_stack)
        for l, dec_layer in enumerate(self.layer_stack):
            scale = 1.
            if self.training:
                drop_level = (l+1.) * self.layer_drop / nl
                if random.random() < drop_level: continue
                scale = 1. / (1.-drop_level)
                
            dec_output = dec_layer(
                dec_output, enc_output, slf_mask=slf_mask,
                dec_enc_mask=dec_enc_mask, scale=scale)
                        
        dec_output = self.layer_norm(dec_output)
        dec_output = self.project(dec_output)
        
        return dec_output

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_vocab=1000, d_input=40, d_model=512, d_inner=2048,
            n_enc=8, n_enc_head=8, n_dec=4, n_dec_head=8, d_k=64,
            use_cnn=False, freq_kn=3, freq_std=2,
            dropout=0.1, emb_drop=0., enc_drop=0.0, dec_drop=0.0,
            shared_kv=False, shared_emb=False, attn_mode=0):

        super().__init__()

        self.encoder = Encoder(
            d_input=d_input, d_model=d_model, d_inner=d_inner,
            n_layers=n_enc, n_head=n_enc_head, d_k=d_k,
            use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std,
            dropout=dropout, layer_drop=enc_drop,
            shared_kv=shared_kv, attn_mode=attn_mode)

        self.decoder = Decoder(
            n_vocab, d_model=d_model, d_inner=d_inner,
            n_layers=n_dec, n_head=n_dec_head, d_k=d_k,
            dropout=dropout, emb_drop=emb_drop, layer_drop=dec_drop,
            shared_emb=shared_emb)

    def forward(self, src_seq, src_mask, tgt_seq):
        enc_output, src_mask = self.encoder(src_seq, src_mask)
        dec_output = self.decoder(tgt_seq, enc_output, src_mask)
        
        return dec_output.view(-1, dec_output.size(2))
        
    def encode(self, src_seq, src_mask):
        return self.encoder(src_seq, src_mask)

    def decode(self, enc_output, src_mask, tgt_seq):
        dec_output = self.decoder(tgt_seq, enc_output, src_mask)
        dec_output = dec_output[:,-1,:].squeeze(1)
        return torch.log_softmax(dec_output, -1)

