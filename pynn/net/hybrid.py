# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import XavierLinear
from .s2s_conformer import Encoder, Decoder

class Hybrid(nn.Module):
    def __init__(self, d_input, n_classes, d_enc=256, d_inner=0, n_enc=4, n_kernel=31,
                 d_dec=320, n_dec=2, n_head=8, d_project=0, shared_emb=True,
                 dropout=0.1, emb_drop=0.1, enc_drop=0., dec_drop=0.,
                 time_ds=1, use_cnn=False, freq_kn=3, freq_std=2):
        super().__init__()

        d_inner = d_enc*4 if d_inner == 0 else d_inner
        self.encoder = Encoder(d_input, d_enc, d_inner, n_enc, n_head, n_kernel=n_kernel,
                            dropout=dropout, layer_drop=enc_drop,
                            time_ds=time_ds, use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std)
        self.decoder = Decoder(n_classes, d_dec, d_inner, n_dec, d_enc, n_head, shared_emb=shared_emb,
                            dropout=dropout, emb_drop=emb_drop, layer_drop=dec_drop)

        d_project = d_enc if d_project==0 else d_project
        self.project = None if d_project==d_enc else XavierLinear(d_enc, d_project)
        self.output = nn.Linear(d_project, n_classes, bias=False)

    def attend(self, src_seq, src_mask, tgt_seq):
        enc_out, src_mask = self.encoder(src_seq, src_mask)[0:2]
        logit, attn = self.decoder(tgt_seq, enc_out, src_mask)[0:2]
        return logit, attn, src_mask

    def forward(self, src_seq, src_mask, tgt_seq, encoding=True, mode=0):
        if mode == 1:
            return self.forward_s2s(src_seq, src_mask, tgt_seq, encoding)

        if encoding:
            enc_out, src_mask = self.encoder(src_seq, src_mask)[0:2]
        else:
            enc_out = src_seq
        dec_out = self.decoder(tgt_seq, enc_out, src_mask)[0]
        ctc_out = self.project(enc_out) if self.project is not None else enc_out
        ctc_out = self.output(ctc_out)
        return dec_out, ctc_out, enc_out, src_mask

    def forward_s2s(self, src_seq, src_mask, tgt_seq, encoding=True):
        if encoding:
            enc_out, src_mask = self.encoder(src_seq, src_mask)[0:2]
        else:
            enc_out = src_seq
        dec_out = self.decoder(tgt_seq, enc_out, src_mask)[0]
        return dec_out, enc_out, src_mask

    def forward_ctc(self, src, mask=None, encoding=True):
        if encoding:
            out, mask = self.encoder(src, mask)[0:2]
        else:
            out = src
        out = self.project(out) if self.project is not None else out
        out = self.output(out)
        return out, mask
        
    def encode(self, src_seq, src_mask, hid=None):
        return self.encoder(src_seq, src_mask, hid)

    def decode(self, enc_out, enc_mask, tgt_seq, hid=None):
        logit, _, hid = self.decoder(tgt_seq, enc_out, enc_mask, hid)
        logit = logit[:,-1,:].squeeze(1)
        return torch.log_softmax(logit, -1), hid

    def decode_s2s(self, enc_out, enc_mask, tgt_seq):
        logit = self.decoder(tgt_seq, enc_out, enc_mask)[0]
        return torch.log_softmax(logit, -1)

    def decode_ctc(self, enc_out, enc_mask):
        logit = self.project(enc_out) if self.project is not None else enc_out
        logit = self.output(logit)
        return torch.log_softmax(logit, -1) 
