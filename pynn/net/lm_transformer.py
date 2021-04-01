# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import XavierLinear
from .attn import PositionalEmbedding
from .transformer_layer import EncoderLayer

class TransformerLM(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, d_inner=0, d_emb=0, d_project=0, n_head=8,
            rel_pos=False, shared_emb=False, dropout=0.2, emb_drop=0., layer_drop=0.):
        super().__init__()

        # Define layers
        d_emb = d_model if d_emb==0 else d_emb
        self.emb = nn.Embedding(n_vocab, d_emb, padding_idx=0)
        self.pos = PositionalEmbedding(d_emb)
        self.scale = d_emb**0.5
        self.emb_drop = nn.Dropout(emb_drop)
        self.rel_pos = rel_pos
        self.emb_prj = None if d_emb==d_model else XavierLinear(d_emb, d_model)
        
        d_inner = d_model*4 if d_inner==0 else d_inner
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, dropout, rel_pos=rel_pos)
            for _ in range(n_layer)])
        self.layer_drop = layer_drop

        d_project = d_model if d_project==0 else d_project
        self.project = None if d_project==d_model else XavierLinear(d_model, d_project)
        self.output = nn.Linear(d_project, n_vocab, bias=True)
        #nn.init.xavier_normal_(self.emb.weight)
        if shared_emb: self.emb.weight = self.output.weight

    def forward(self, inputs):
        out = self.emb(inputs)
        if self.rel_pos:
            pos_emb = self.pos.embed(inputs.gt(0))
        else:
            pos_seq = torch.arange(0, out.size(1), device=out.device, dtype=out.dtype)
            pos_emb = self.pos(pos_seq, out.size(0))
            out = out + pos_emb
        out = self.emb_drop(out)
        out = self.emb_prj(out) if self.emb_prj is not None else out

        lt = inputs.size(1)
        # -- Prepare masks
        slf_mask = inputs.eq(0).unsqueeze(1).expand(-1, lt, -1)
        tri_mask = torch.ones((lt, lt), device=out.device, dtype=torch.uint8)
        tri_mask = torch.triu(tri_mask, diagonal=1)
        tri_mask = tri_mask.unsqueeze(0).expand(inputs.size(0), -1, -1)
        slf_mask = (slf_mask + tri_mask).gt(0)

        nl = len(self.layer_stack)
        for l, enc_layer in enumerate(self.layer_stack):
            scale = 1.
            if self.training:
                drop_level = (l+1.) * self.layer_drop / nl
                if random.random() < drop_level: continue
                scale = 1. / (1.-drop_level)
            
            out = (out, pos_emb) if self.rel_pos else out
            out = enc_layer(out, slf_mask=slf_mask, scale=scale)

        out = self.project(out) if self.project is not None else out
        out = self.output(out)

        return out
        
    def decode(self, inputs):
        logits = self.forward(inputs)
        logits = logits[:,-1,:].squeeze(1)
        return torch.log_softmax(logits, -1)
