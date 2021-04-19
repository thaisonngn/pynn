
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .transformer_layer import DecoderLayer, PositionwiseFF
from .attn import MultiHeadedAttention

class AttentionMemory(nn.Module):
    def __init__(self, d_model, size_memory):
        super().__init__()

        self.linear = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.temperature = np.power(d_model, 0.5)

        self.layer_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(size_memory+1,elementwise_affine=False)

    def forward(self, dec_output, enc_out_mem_mean, mem_attn_out=None):
        dec_output = self.layer_norm(dec_output)

        # attention over dictionary entries
        q = self.linear(dec_output).unsqueeze(2)  # b x l_tar x 1 x d_model
        k = self.linear2(enc_out_mem_mean)  # n_st x d_model

        attn = torch.matmul(q, k.transpose(1, 0))[:, :, 0]  # b x l_tar x n_st
        attn = attn / self.temperature
        attn = self.norm(attn.view(-1,attn.shape[-1])).view(*attn.shape)
        if not mem_attn_out is None:  # skip connection
            attn = (attn + mem_attn_out) / 2

        mem_attn_out = attn  # for loss computation and next decoder layer
        return mem_attn_out

def get_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    len_q = seq_q.size(1)
    padding_mask = seq_k.logical_not()
    return padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

class AttentionMemoryEntry(nn.Module):
    def __init__(self, n_head, d_model, d_inner, dropout, version_gate=0):
        super().__init__()

        self.version_gate = version_gate

        self.st_attn = MultiHeadedAttention(n_head, d_model, dropout, residual=True)

        self.pos_ffn = PositionwiseFF(d_model, d_inner, dropout, residual=True)
        self.pos_ffn2 = PositionwiseFF(d_model, d_inner, dropout, residual=True)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

    def forward(self, dec_output, tgt_mask, mem_attn_out, enc_out_mem, tgt_emb_mem, tgt_mask_mem):
        dec_output = self.norms[0](dec_output)

        samples = mem_attn_out.argmax(-1)-1  # b x l_tar

        b, l_tar = samples.shape
        indices = torch.arange(b, device=samples.device).view(-1, 1).expand(-1, l_tar).reshape(-1, 1)  # b*l_tar x 1

        tmp = torch.cat([indices, samples.view(-1, 1)], 1)

        # filter -1´s
        mask = tmp[:,1].ne(-1)
        if mask.any():
            tmp = tmp[mask]

            # unique
            tmp, inv = torch.unique(tmp, return_inverse=True, dim=0)
            indices = tmp[:, 0]
            samples2 = tmp[:, 1]

            dec_dec_mask = get_key_pad_mask(seq_k=tgt_mask_mem[samples2], seq_q=tgt_mask[indices])

            st_attn, _ = self.st_attn(dec_output[indices], enc_out_mem[samples2], mask=dec_dec_mask,
                                      value=tgt_emb_mem[samples2]) # ? x l_tar x d_model

            #printms(0, st_attn)

            st_attn = self.pos_ffn(st_attn)
            st_attn = self.norms[1](st_attn)

            # undo unique
            st_attn = st_attn.gather(0, inv.view(-1, 1, 1).expand(-1, st_attn.shape[1],
                                                                  st_attn.shape[2]))  # ?? x l_tar x d_model

            # undo filter -1´s
            mask2 = torch.arange(b*l_tar, device=mask.device)[mask].unsqueeze(-1).unsqueeze(-1).expand(-1,l_tar,st_attn.shape[-1])
            st_attn = torch.zeros(b*l_tar, l_tar, st_attn.shape[-1], device=st_attn.device, dtype=st_attn.dtype).\
                scatter_(0,mask2,st_attn) # b*l_tar x l_tar x d_model

            st_attn = st_attn.gather(1,
                                     torch.arange(l_tar, device=samples2.device).view(1, -1, 1).expand(b, -1,
                                                                                                      st_attn.shape[
                                                                                                          -1]).reshape(
                                         b * l_tar, 1, -1)
                                     ).view(b, l_tar, -1)  # b x l_tar x d_model

            if self.version_gate==0:
                dec_output = dec_output + st_attn
            elif self.version_gate==1:
                raise NotImplementedError("Implementation to be checked")
                gate = F.softmax(mem_attn_out.to(torch.float32),-1).view(-1,mem_attn_out.shape[-1])[samples.view(-1)]\
                    .reshape((*st_attn.shape[:2],1)).to(st_attn.dtype)
                dec_output = dec_output + gate * st_attn

        dec_output = self.pos_ffn2(dec_output)
        return dec_output

class DecoderLayerMemory(DecoderLayer):
    def __init__(self, d_model, d_inner, n_head, dropout=0.1, n_enc_head=0, rel_pos=False,
                 size_memory=200, version_gate=0):
        super().__init__(d_model, d_inner, n_head, dropout=dropout, n_enc_head=n_enc_head, rel_pos=rel_pos)

        self.attentionMemory = AttentionMemory(d_model, size_memory)
        self.attentionMemoryEntry = AttentionMemoryEntry(n_head, d_model, d_inner, dropout, version_gate=version_gate)

    def forward(self, dec_inp, enc_out, slf_mask=None, dec_enc_mask=None, scale=1.,
                enc_out_mem_mean=None, tgt_mask=None, mem_attn_out=None, enc_out_mem=None, tgt_emb_mem=None, tgt_mask_mem=None):
        dec_output = self.slf_attn(dec_inp, mask=slf_mask, scale=scale)[0]
        dec_output, attn = self.enc_attn(dec_output, enc_out, mask=dec_enc_mask, scale=scale)
        dec_output = self.pos_ffn(dec_output, scale=scale)

        mem_attn_out = self.attentionMemory(dec_output, enc_out_mem_mean, mem_attn_out)
        dec_output = self.attentionMemoryEntry(dec_output, tgt_mask, mem_attn_out, enc_out_mem, tgt_emb_mem, tgt_mask_mem)

        return dec_output, mem_attn_out

def printms(s,x):
    print(s,torch.mean(x).item(),torch.std(x).item())