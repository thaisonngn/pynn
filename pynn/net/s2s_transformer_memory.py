
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import XavierLinear, Swish
from .attn import PositionalEmbedding
from .s2s_transformer import Encoder, Decoder
from .memory_layer import DecoderLayerMemory

class DecoderMemory(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, n_head, d_inner,
                 rel_pos=False, dropout=0.1, emb_drop=0., layer_drop=0., shared_emb=True,
                 size_memory=200, version_gate=0):

        super().__init__()

        self.emb = nn.Embedding(n_vocab, d_model, padding_idx=0)
        self.pos = PositionalEmbedding(d_model)
        self.scale = d_model ** 0.5
        self.emb_drop = nn.Dropout(emb_drop)
        self.rel_pos = rel_pos

        self.layer_stack = nn.ModuleList([
            DecoderLayerMemory(d_model, d_inner, n_head, dropout, rel_pos=rel_pos,
                               size_memory=size_memory, version_gate=version_gate)
            for _ in range(n_layer)])

        self.output = nn.Linear(d_model, n_vocab, bias=True)
        # nn.init.xavier_normal_(self.project.weight)
        if shared_emb: self.emb.weight = self.output.weight

        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_drop = layer_drop

    def forward(self, tgt_seq, enc_out, enc_mask, tgt_emb_mem, tgt_mask_mem, enc_out_mem, enc_out_mem_mean):
        # -- Forward
        dec_out = self.emb(tgt_seq) * self.scale
        tgt_mask = tgt_seq.gt(0)

        if self.rel_pos:
            pos_emb = self.pos.embed(tgt_mask)
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

        nl = len(self.layer_stack)
        mem_attn_out = None
        mem_attn_outs = []
        for l, dec_layer in enumerate(self.layer_stack):
            scale = 1.
            if self.training:
                drop_level = (l + 1.) * self.layer_drop / nl
                if random.random() < drop_level: continue
                scale = 1. / (1. - drop_level)

            dec_out = (dec_out, pos_emb) if self.rel_pos else dec_out
            dec_out, mem_attn_out = dec_layer(
                dec_out, enc_out, slf_mask=slf_mask,
                dec_enc_mask=attn_mask, scale=scale,
                enc_out_mem_mean=enc_out_mem_mean, tgt_mask=tgt_mask, mem_attn_out=mem_attn_out,
                enc_out_mem=enc_out_mem, tgt_emb_mem=tgt_emb_mem, tgt_mask_mem=tgt_mask_mem)

            mem_attn_outs.append(mem_attn_out)

        dec_out = self.layer_norm(dec_out)
        dec_out = self.output(dec_out)

        return dec_out, mem_attn_outs

class TransformerMemory(nn.Module):
    def __init__(
            self,
            n_vocab=1000, n_emb=0, d_input=40, d_model=512, d_inner=2048,
            n_enc=8, n_enc_head=8, n_dec=4, n_dec_head=8,
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2,
            dropout=0.1, emb_drop=0., enc_drop=0.0, dec_drop=0.0,
            shared_emb=False, rel_pos=False,
            size_memory=200, version_gate=0, n_enc_mem=8):

        super().__init__()

        self.encoder = Encoder(
            d_input=d_input, d_model=d_model, d_inner=d_inner,
            n_layer=n_enc, n_head=n_enc_head, rel_pos=rel_pos,
            embedding=(n_emb > 0), emb_vocab=n_emb, emb_drop=emb_drop,
            time_ds=time_ds, use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std,
            dropout=dropout, layer_drop=enc_drop)

        self.encoder_mem = Encoder(
            d_input=d_model, d_model=d_model, d_inner=d_inner,
            n_layer=n_enc_mem, n_head=n_enc_head, rel_pos=rel_pos,
            embedding=False, emb_vocab=n_vocab, emb_drop=emb_drop,
            time_ds=1, use_cnn=False, dropout=dropout, layer_drop=enc_drop)

        self.decoder = Decoder(
            n_vocab, d_model=d_model, d_inner=d_inner, n_layer=n_dec,
            n_head=n_dec_head, shared_emb=shared_emb, rel_pos=False,
            dropout=dropout, emb_drop=emb_drop, layer_drop=dec_drop)

        self.decoder_mem = DecoderMemory(
            n_vocab, d_model=d_model, d_inner=d_inner, n_layer=n_dec,
            n_head=n_dec_head, shared_emb=shared_emb, rel_pos=False,
            dropout=dropout, emb_drop=emb_drop, layer_drop=0,
            size_memory=size_memory, version_gate=version_gate)

        self.project = nn.Linear(2*n_dec,2)
        self.n_vocab = n_vocab

    def forward(self, src_seq, src_mask, tgt_seq, tgt_ids_mem, label_gate=None, gold=None, encoding=True, enc_out=None):
        if encoding:
            enc_out = self.encode(src_seq, src_mask, tgt_ids_mem)

        dec_out, mem_attn_outs = self.decode(tgt_seq, enc_out, label_gate, gold, inference=False)
        return dec_out, mem_attn_outs, enc_out

    def encode(self, src_seq, src_mask, tgt_ids_mem):
        enc_out, enc_mask = self.encoder(src_seq, src_mask)

        # generate tgt and gold sequence in the memory
        tgt_emb_mem = self.decoder_mem.emb(tgt_ids_mem)
        tgt_mask_mem = tgt_ids_mem.ne(0)

        if self.decoder_mem.rel_pos:
            raise NotImplementedError
            #pos_emb = self.pos.embed(tgt_emb_mem)
        else:
            pos_seq = torch.arange(0, tgt_emb_mem.size(1), device=tgt_emb_mem.device, dtype=tgt_emb_mem.dtype)
            pos_emb = self.decoder_mem.pos(pos_seq, tgt_emb_mem.size(0))
            tgt_emb_mem = tgt_emb_mem * self.decoder_mem.scale + pos_emb
        tgt_seq_mem = self.decoder_mem.emb_drop(tgt_emb_mem)

        # encode tgt seq from the memory
        enc_out_mem = self.encoder_mem(tgt_seq_mem, tgt_mask_mem)[0]

        # calc mean
        enc_out_mem[tgt_ids_mem.eq(0)] = 0
        enc_out_mem_mean = enc_out_mem.sum(1) / (tgt_mask_mem.sum(1, keepdims=True)) # n_mem x d_model

        return enc_out, enc_mask, tgt_emb_mem, tgt_mask_mem, enc_out_mem, enc_out_mem_mean

    def decode(self, tgt_seq, enc_out, label_gate=None, gold=None, inference=True):
        enc_out, enc_mask, tgt_emb_mem, tgt_mask_mem, enc_out_mem, enc_out_mem_mean = enc_out

        dec_out_orig = self.decoder(tgt_seq, enc_out, enc_mask)[0]
        dec_out_mem, mem_attn_outs = self.decoder_mem(tgt_seq, enc_out, enc_mask, tgt_emb_mem, tgt_mask_mem, enc_out_mem, enc_out_mem_mean)

        fp16 = dec_out_orig.dtype == torch.float16

        dec_out_orig = F.softmax(dec_out_orig.to(torch.float32),-1)
        dec_out_mem = F.softmax(dec_out_mem.to(torch.float32), -1)

        if not label_gate is None:
            mask = gold.gt(2)

            dec_out_orig = self.noise_permute(dec_out_orig, gold, label_gate.eq(1) & mask)
            dec_out_mem = self.noise_permute(dec_out_mem, gold, label_gate.eq(0) & mask)

        gates = torch.cat([F.softmax(a[0].to(torch.float32), -1).detach() for a in mem_attn_outs], -1)
        gates = F.softmax(self.project(gates if not fp16 else gates.to(torch.float16)).to(torch.float32), -1)

        dec_output = gates[:, :, 0:1] * dec_out_orig + gates[:, :, 1:2] * dec_out_mem

        if not inference:
            return dec_output, mem_attn_outs
        else:
            dec_output = dec_output[:, -1, :].squeeze(1)
            return torch.log(dec_output)

    def noise_permute(self, pred, gold_idx, mask, noise_prob=0.5, detach=True):
        b, l_tgt, _ = pred.shape
        pred = pred.view(b * l_tgt, -1)
        gold_idx = gold_idx.reshape(b * l_tgt)
        mask = mask.view(-1)

        mask = torch.empty((b * l_tgt), dtype=torch.long, device=pred.device).bernoulli_(noise_prob).eq(1) & mask
        ind = torch.arange(b * l_tgt, device=pred.device)[mask]
        rpl_idx = torch.randint(0, self.n_vocab - 1, (b * l_tgt,), device=pred.device)

        rpl_idx_small = rpl_idx.gather(0, ind).unsqueeze(-1)
        gold_idx_small = gold_idx.gather(0, ind).unsqueeze(-1)

        ind2 = ind.unsqueeze(-1).expand(-1, pred.shape[-1])

        pred_small = pred.gather(0, ind2)

        probs_gold = pred_small.gather(1, gold_idx_small)
        probs_rpl = pred_small.gather(1, rpl_idx_small)
        if detach:
            probs_gold = probs_gold.detach()
            probs_rpl = probs_rpl.detach()
        pred_small = pred_small.scatter(1, rpl_idx_small, probs_gold)
        pred_small = pred_small.scatter(1, gold_idx_small, probs_rpl)

        pred = pred.scatter(0, ind2, pred_small)

        pred = pred.view(b, l_tgt, -1)
        return pred