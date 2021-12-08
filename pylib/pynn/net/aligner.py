# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s2s_transformer import Encoder as AttnEncoder

class AttnAligner(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, rel_pos=False, dropout=0.2):
        super().__init__()
        self.p_size = n_vocab
        self.encoder = AttnEncoder(n_vocab, d_model, n_layer, 8, d_model*4,
                                   rel_pos=rel_pos, dropout=dropout)
        self.project = nn.Linear(d_model, self.p_size, bias=False)

    def align(self, s2s_model, inputs, masks, tgt):
        attn, masks = s2s_model.attend(inputs, masks, tgt)[1:3]
        attn = attn[:, 0, 1:-1, :].contiguous()
        tgt = tgt[:, 1:-1].contiguous()
        alg = self.forward(attn, masks, tgt)
        return attn, masks, alg, tgt

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
