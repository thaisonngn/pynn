# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import freeze_module
from .attn import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers, unidirect=False, incl_win=0,
            dropout=0.2, time_ds=1, use_cnn=False, freq_kn=3, freq_std=2):
        super().__init__()

        self.time_ds = time_ds
        if use_cnn:
            cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)),
                   nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std))]
            self.cnn = nn.Sequential(*cnn)
            input_size = ((((input_size - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1)*32
        else:
            self.cnn = None

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers,
                        bidirectional=(not unidirect), bias=False, dropout=dropout, batch_first=True)
        self.unidirect = unidirect
        self.incl_win = incl_win

    def rnn_fwd(self, seq, mask, hid):
        if mask is not None:
            lengths = mask.sum(-1); #lengths[0] = mask.size(1)
            seq = pack_padded_sequence(seq, lengths, batch_first=True)    
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
        #out *= mask.unsqueeze(-1).type(out.dtype)

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
    def __init__(self, output_size, hidden_size, layers, n_head=8, lm=True, shared_emb=True,
            dropout=0.2, emb_drop=0.):
        super().__init__()

        # Keep for reference
        self.lm = lm

        # Define layers
        self.emb = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.scale = hidden_size**0.5 if shared_emb else 1.
        self.emb_drop = nn.Dropout(emb_drop)

        d_k = hidden_size // n_head
        self.attn = MultiHeadAttention(n_head, hidden_size, d_k, dropout=dropout, norm=True, res=False)

        dropout = (0 if layers == 1 else dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, layers, dropout=dropout, batch_first=True)

        self.project = nn.Linear(hidden_size, output_size, bias=True)
        #nn.init.xavier_normal_(self.project.weight)
        if shared_emb: self.emb.weight = self.project.weight

    def forward(self, dec_seq, enc_out, enc_mask, hid=None):
        dec_emb = self.emb(dec_seq) * self.scale
        dec_emb = self.emb_drop(dec_emb)

        if dec_seq.size(0) > 1 and dec_seq.size(1) > 1:
            lengths = dec_seq.gt(0).sum(-1); #lengths[0] = dec_seq.size(1)
            dec_in = pack_padded_sequence(dec_emb, lengths, batch_first=True, enforce_sorted=False)
            dec_out, hid = self.lstm(dec_in, hid)
            dec_out = pad_packed_sequence(dec_out, batch_first=True)[0]
        else:
            dec_out, hid = self.lstm(dec_emb, hid)

        lt = dec_out.size(1)
        attn_mask = enc_mask.eq(0).unsqueeze(1).expand(-1, lt, -1)
        
        context, attn = self.attn(dec_out, enc_out, mask=attn_mask)
        context = (context + dec_out) if self.lm else context
        output = self.project(context)

        return output, attn, hid

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_enc=5, n_dec=2,
            n_head=8, unidirect=False, incl_win=0, lm=True, shared_emb=True, dropout=0.2, emb_drop=0.,
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(input_size, hidden_size, n_enc, dropout=dropout,
                            unidirect=unidirect, incl_win=incl_win, time_ds=time_ds,
                            use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std)
        self.decoder = Decoder(output_size, hidden_size, n_dec, n_head,
                            lm=lm, shared_emb=shared_emb, dropout=dropout, emb_drop=emb_drop)

    def freeze(self, mode=0):
        if mode == 1:
            freeze_module(self.encoder)
            print("freeze the encoder")

    def attend(self, src_seq, src_mask, tgt_seq):
        enc_out, src_mask = self.encoder(src_seq, src_mask)[0:2]
        logit, attn = self.decoder(tgt_seq, enc_out, src_mask)[0:2]
        return logit, attn, src_mask

    def forward(self, src_seq, src_mask, tgt_seq, encoding=True):
        if encoding:
            enc_out, src_mask = self.encoder(src_seq, src_mask)[0:2]
        else:
            enc_out = src_seq

        dec_out = self.decoder(tgt_seq, enc_out, src_mask)[0]
        #return logit.view(-1, logit.size(2))
        return dec_out, enc_out, src_mask

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

    def converage(self, enc_out, masks, targets, alpha=0.05):
        attn = self.decoder(targets, enc_out, masks)[1]
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
