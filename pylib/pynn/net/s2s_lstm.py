# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import freeze_module
from . import XavierLinear, Swish
from .rnn import LSTM
from .attn import MultiHeadedAttention

class Encoder(nn.Module):
    def __init__(self, d_input, d_model, n_layer, unidirect=False, incl_win=0,
            dropout=0.2, dropconnect=0., time_ds=1, use_cnn=False, freq_kn=3, freq_std=2, pack=True):
        super().__init__()

        self.time_ds = time_ds
        if use_cnn:
            cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)), Swish(),
                   nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)), Swish()]
            self.cnn = nn.Sequential(*cnn)
            d_input = ((((d_input - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1)*32
        else:
            self.cnn = None

        self.rnn = LSTM(input_size=d_input, hidden_size=d_model, num_layers=n_layer, batch_first=True,
                        bidirectional=(not unidirect), bias=False, dropout=dropout, dropconnect=dropconnect)
        self.unidirect = unidirect
        self.incl_win = incl_win
        self.pack = pack

    def rnn_fwd(self, seq, mask, hid):
        if self.pack and mask is not None:
            lengths = mask.sum(-1); #lengths[0] = mask.size(1)
            seq = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
            seq, hid = self.rnn(seq, hid)
            seq = pad_packed_sequence(seq, batch_first=True)[0]
        else:
            seq, hid = self.rnn(seq)

        return seq, hid

    def rnn_fwd_incl(self, seq, mask, hid=None):
        win, time_len = self.incl_win, seq.size(1)
        out = []
        for i in range((time_len-1) // win + 1):
            s, e = win*i, min(time_len, win*(i+1))
            src = seq[:, s:e, :]
            enc, hid = self.rnn(src, hid)
            out.append(enc)
        out = torch.cat(out, dim=1)

        return out, hid

    def forward(self, seq, mask=None, hid=None):
        if self.time_ds > 1:
            ds = self.time_ds
            l = ((seq.size(1) - 3) // ds) * ds
            seq = seq[:, :l, :]
            seq = seq.view(seq.size(0), -1, seq.size(2)*ds)
            if mask is not None: mask = mask[:, 0:seq.size(1)*ds:ds]

        if self.cnn is not None:
            seq = self.cnn(seq.unsqueeze(1))
            seq = seq.permute(0, 2, 1, 3).contiguous()
            seq = seq.view(seq.size(0), seq.size(1), -1)
            if mask is not None: mask = mask[:, 0:seq.size(1)*4:4]

        seq, hid = self.rnn_fwd(seq, mask, hid) \
                if self.incl_win==0 else self.rnn_fwd_incl(seq, mask, hid)
 
        if not self.unidirect:
            hidden_size = seq.size(2) // 2
            seq = seq[:, :, :hidden_size] + seq[:, :, hidden_size:]
        
        return seq, mask, hid

        
class Decoder(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, d_enc, d_emb=0, d_project=0,
            n_head=8, shared_emb=True, dropout=0.2, dropconnect=0., emb_drop=0., pack=True):
        super().__init__()

        # Define layers
        d_emb = d_model if d_emb==0 else d_emb
        self.emb = nn.Embedding(n_vocab, d_emb, padding_idx=0)
        self.emb_drop = nn.Dropout(emb_drop)
        self.scale = d_emb**0.5

        self.attn = MultiHeadedAttention(n_head, d_model, dropout, residual=False)
        dropout = (0 if n_layer == 1 else dropout)
        self.lstm = LSTM(d_emb, d_model, n_layer, batch_first=True, dropout=dropout, dropconnect=dropconnect)
        self.transform = None if d_model==d_enc else XavierLinear(d_enc, d_model)
        d_project = d_model if d_project==0 else d_project
        self.project = None if d_project==d_model else XavierLinear(d_model, d_project)
        self.output = nn.Linear(d_project, n_vocab, bias=False)
        #nn.init.xavier_normal_(self.project.weight)
        if shared_emb: self.emb.weight = self.output.weight
        self.pack = pack

    def forward(self, dec_seq, enc_out, enc_mask, hid=None):
        dec_emb = self.emb(dec_seq) * self.scale
        dec_emb = self.emb_drop(dec_emb)
        
        if self.pack and dec_seq.size(0) > 1 and dec_seq.size(1) > 1:
            lengths = dec_seq.gt(0).sum(-1); #lengths[0] = dec_seq.size(1)
            dec_in = pack_padded_sequence(dec_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            dec_out, hid = self.lstm(dec_in, hid)
            dec_out = pad_packed_sequence(dec_out, batch_first=True)[0]
        else:
            dec_out, hid = self.lstm(dec_emb, hid)
        enc_out = self.transform(enc_out) if self.transform is not None else enc_out
        lt = dec_out.size(1)
        attn_mask = enc_mask.eq(0).unsqueeze(1).expand(-1, lt, -1)
         
        context, attn = self.attn(dec_out, enc_out, mask=attn_mask)
        out = context + dec_out
        out = self.project(out) if self.project is not None else out
        out = self.output(out)

        return out, attn, hid

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
        self.decoder = Decoder(n_vocab, d_dec, n_dec, d_enc,
                            n_head=n_head, d_emb=d_emb, d_project=d_project, shared_emb=shared_emb,
                            dropout=dec_dropout, dropconnect=dec_dropconnect, emb_drop=emb_drop, pack=pack)

    def freeze(self, mode=0):
        if mode == 1:
            freeze_module(self.encoder)
            print("freeze the encoder")

    def attend(self, src_seq, src_mask, tgt_seq):
        enc_out, enc_mask = self.encoder(src_seq, src_mask)[0:2]
        logit, attn = self.decoder(tgt_seq, enc_out, enc_mask)[0:2]
        return logit, attn, enc_mask

    def forward(self, src_seq, src_mask, tgt_seq, encoding=True):
        if encoding:
            enc_out, enc_mask = self.encoder(src_seq, src_mask)[0:2]
        else:
            enc_out, enc_mask = src_seq, src_mask

        dec_out = self.decoder(tgt_seq, enc_out, enc_mask)[0]
        #return logit.view(-1, logit.size(2))
        return dec_out, enc_out, enc_mask

    def encode(self, src_seq, src_mask, hid=None):
        return self.encoder(src_seq, src_mask, hid)

    def decode(self, enc_out, enc_mask, tgt_seq, hid=None):
        logit, attn, hid = self.decoder(tgt_seq, enc_out, enc_mask, hid)
        logit = logit[:,-1,:].squeeze(1)
        return torch.log_softmax(logit, -1), hid

    def get_attn(self, enc_out, enc_mask, tgt_seq):
        return self.decoder(tgt_seq, enc_out, enc_mask)[1]
        
    def get_logit(self, enc_out, enc_mask, tgt_seq):
        logit = self.decoder(tgt_seq, enc_out, enc_mask)[0]
        return torch.log_softmax(logit, -1)
