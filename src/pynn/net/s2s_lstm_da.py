# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import freeze_module
from .s2s_transformer import Encoder as AttnEncoder
from .s2s_lstm import Encoder, Decoder

class AttnAligner(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, dropout=0.2):
        super().__init__()
        self.p_size = n_vocab
        self.encoder = AttnEncoder(n_vocab, d_model, n_layer, 8, d_model*4, dropout=dropout)
        self.project = nn.Linear(d_model, self.p_size, bias=False)

    def forward(self, attn, mask, tgt):
        bs, ls, ps = tgt.size(0), tgt.size(1), self.p_size
        idx = torch.arange(0, bs*ps, ps, device=tgt.device)
        idx = idx.unsqueeze(1).expand(-1, ls) + tgt
        idx = idx.view(-1)
        ll = attn.size(2)
        alg = torch.zeros(ps*bs, ll, device=tgt.device).type(attn.dtype)
        #alg.fill_(1. / ps)
        #attn = attn.exp().log_softmax(dim=1)
        attn = attn.view(-1, ll)
        alg.index_put_([idx], attn, accumulate=True)
        alg = alg.view(bs, ps, -1).permute(0, 2, 1)
        alg.index_fill_(2, torch.LongTensor([0]).to(tgt.device), 0)
        #alg.log_()
        #alg = alg / alg.sum(-1, keepdim=True)

        alg = self.project(self.encoder(alg, mask)[0])
        return alg

class Seq2Seq(nn.Module):
    def __init__(self, n_vocab=1000, d_input=40, d_enc=320, n_enc=6, d_dec=320, n_dec=2,
            unidirect=False, incl_win=0, d_emb=0, d_project=0, n_head=8, shared_emb=False, 
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2, enc_dropout=0.2, enc_dropconnect=0., 
            dec_dropout=0.1, dec_dropconnect=0., emb_drop=0., pack=True, n_aligner=1):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(d_input, d_enc, n_enc,
                            unidirect=unidirect, incl_win=incl_win, time_ds=time_ds,
                            use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std,
                            dropout=enc_dropout, dropconnect=enc_dropconnect, pack=pack)
        self.decoder = Decoder(n_vocab, d_dec, n_dec, d_enc,
                            n_head=n_head, d_emb=d_emb, d_project=d_project, shared_emb=shared_emb,
                            dropout=dec_dropout, dropconnect=dec_dropconnect, emb_drop=emb_drop, pack=pack)
        self.aligner = AttnAligner(n_vocab, 512, n_aligner)

    def freeze(self):
        freeze_module(self.encoder)
        freeze_module(self.decoder)
        print("Freeze the encoder and decoder")

    def forward(self, src_seq, src_mask, tgt_seq, encoding=True, mode=0):
        if encoding:
            enc_out, enc_mask = self.encoder(src_seq, src_mask)[0:2]
        else:
            enc_out, enc_mask = src_seq, src_mask

        dec_out, attn = self.decoder(tgt_seq, enc_out, enc_mask)[0:2]
        if mode == 1: return  dec_out, enc_out, enc_mask

        attn = attn[:, 0, 1:, :].contiguous()
        tgt_seq = tgt_seq[:, 1:].contiguous()
        alg_out = self.aligner(attn, enc_mask, tgt_seq)

        return dec_out, alg_out, enc_out, enc_mask

    def encode(self, src_seq, src_mask, hid=None):
        return self.encoder(src_seq, src_mask, hid)

    def decode(self, enc_out, enc_mask, tgt_seq, hid=None):
        logit, attn, hid = self.decoder(tgt_seq, enc_out, enc_mask, hid)
        logit = logit[:,-1,:].squeeze(1)
        return torch.log_softmax(logit, -1), attn, hid

    def decode_s2s(self, enc_out, enc_mask, tgt_seq):
        logit = self.decoder(tgt_seq, enc_out, enc_mask)[0]
        return torch.log_softmax(logit, -1)

    def decode_ctc(self, enc_out, enc_mask, tgt_seq):
        dec_out, attn = self.decoder(tgt_seq, enc_out, enc_mask)[0:2]
        attn = attn[:, 0, 1:-1, :].contiguous()
        tgt_seq = tgt_seq[:, 1:-1].contiguous()
        logit = self.aligner(attn, enc_mask, tgt_seq)

        return torch.log_softmax(logit, -1)

    def attend(self, src_seq, src_mask, tgt_seq):
        enc_out, enc_mask = self.encoder(src_seq, src_mask)[0:2]
        logit, attn = self.decoder(tgt_seq, enc_out, enc_mask)[0:2]
        return logit, attn, enc_mask

    def align(self, s2s_model, inputs, masks, tgt):
        attn, masks = s2s_model.attend(inputs, masks, tgt)[1:3]
        attn = attn[:, 0, 1:-1, :].contiguous()
        tgt = tgt[:, 1:-1].contiguous()
        alg = self.aligner(attn, masks, tgt)
        return attn, masks, alg, tgt
