
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

        self.linear_gate = nn.Linear(d_model, d_model)
        self.linear2_gate = nn.Linear(d_model, d_model)

        self.linear3 = nn.Linear(size_memory, 128)
        self.linear4 = nn.Linear(128, 2)

        self.layer_norm = nn.LayerNorm(d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(size_memory,elementwise_affine=False) for _ in range(2)])

    def forward(self, dec_output, enc_out_mem_mean, mem_attn_out=None):
        dec_output = self.layer_norm(dec_output)

        #printms(0, dec_output)
        #printms(1, dec_output_dic_mean)

        # attention over dictionary entries
        q = self.linear(dec_output).unsqueeze(2)  # b x l_tar x 1 x d_model
        k = self.linear2(enc_out_mem_mean)  # n_st x d_model

        attn = torch.matmul(q, k.transpose(1, 0))[:, :, 0]  # b x l_tar x n_st
        attn = attn / self.temperature
        attn = self.norms[0](attn.view(-1,attn.shape[-1])).view(*attn.shape)
        if not mem_attn_out is None:  # skip connection
            attn = (attn + mem_attn_out[1]) / 2
        #printms(2, attn)

        q = self.linear_gate(dec_output).unsqueeze(2)  # b x l_tar x 1 x d_model
        k = self.linear2_gate(enc_out_mem_mean)  # n_st x d_model
        attn_gate = torch.matmul(q, k.transpose(1, 0))[:, :, 0]  # b x l_tar x n_st
        attn_gate = attn_gate / self.temperature
        attn_gate = self.norms[1](attn_gate.view(-1, attn_gate.shape[-1])).view(*attn_gate.shape)
        if not mem_attn_out is None:  # skip connection
            attn_gate = (attn_gate + mem_attn_out[2]) / 2
        attn_gate_save = attn_gate
        #printms(3, attn_gate)

        attn_gate = torch.sort(attn_gate, descending=True)[0][:, :, :self.linear3.in_features]
        attn_gate = self.linear4(F.relu(self.linear3(attn_gate)))  # b x l_tar x 2
        if not mem_attn_out is None:  # skip connection
            attn_gate = (attn_gate + mem_attn_out[0]) / 2
        #printms(4, attn_gate)

        mem_attn_out = (attn_gate, attn, attn_gate_save)  # for loss computation and next decoder layer
        return mem_attn_out

def get_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    len_q = seq_q.size(1)
    padding_mask = seq_k.logical_not()
    return padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

class AttentionMemoryEntry(nn.Module):
    def __init__(self, n_head, d_model, d_inner, dropout, version_gate):
        super().__init__()

        self.version_gate = version_gate

        self.st_attn = MultiHeadedAttention(n_head, d_model, dropout, residual=True)

        self.pos_ffn = PositionwiseFF(d_model, d_inner, dropout, residual=True)
        self.pos_ffn2 = PositionwiseFF(d_model, d_inner, dropout, residual=True)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

        if self.version_gate==2:
            self.linear = nn.Linear(1,1)

    def forward(self, dec_output, tgt_mask, mem_attn_out, enc_out_mem, tgt_emb_mem, tgt_mask_mem):
        dec_output = self.norms[0](dec_output)

        samples = mem_attn_out[1].argmax(-1)  # b x l_tar

        b, l_tar = samples.shape

        indices = torch.arange(b, device=samples.device).view(-1, 1).expand(-1, l_tar).reshape(-1, 1)  # b*l_tar x 1

        tmp = torch.cat([indices, samples.view(-1, 1)], 1)
        tmp, inv = torch.unique(tmp, return_inverse=True, dim=0)
        indices = tmp[:, 0]
        samples = tmp[:, 1]

        dec_dec_mask = get_key_pad_mask(seq_k=tgt_mask_mem[samples], seq_q=tgt_mask[indices])

        st_attn, _ = self.st_attn(dec_output[indices], enc_out_mem[samples], mask=dec_dec_mask,
                                  value=tgt_emb_mem[samples])
        st_attn = st_attn.gather(0, inv.view(-1, 1, 1).expand(-1, st_attn.shape[1],
                                                              st_attn.shape[2]))  # b*l_tar x l_tar x d_model
        st_attn = st_attn.gather(1,
                                 torch.arange(l_tar, device=samples.device).view(1, -1, 1).expand(b, -1,
                                                                                                  st_attn.shape[
                                                                                                      -1]).reshape(
                                     b * l_tar, 1, -1)
                                 ).view(b, l_tar, -1)  # b x l_tar x d_model

        st_attn = self.pos_ffn(st_attn)

        st_attn = self.norms[1](st_attn)

        if self.version_gate==0:
            dec_output = dec_output + st_attn
        elif self.version_gate==1:
            gate = F.softmax(mem_attn_out[0],-1).detach()
            #print(gate[0,:,1])
            dec_output = dec_output + gate[:, :, 1:2] * st_attn
        elif self.version_gate==2:
            gate = F.softmax(mem_attn_out[0], -1).detach()
            #print(gate[0,:,1])
            gate = F.sigmoid(self.linear(gate[:, :, 1:2]))
            #print(gate[0,:,0])
            dec_output = dec_output + gate[:, :, 1:2] * st_attn

        dec_output = self.pos_ffn2(dec_output)

        return dec_output

    def get_det_attn(self, tgt_seq, attn_dic, dic):  # deterministic memory attention
        #print(tgt_seq[0],tgt_seq.shape)
        #print(attn_dic[0],attn_dic.shape)
        #print(dic[attn_dic[0]],dic[attn_dic[0]].shape)

        b = tgt_seq.shape[0]
        l_tgt = tgt_seq.shape[1]
        l_dic = dic.shape[1]-1

        attn_dic = attn_dic.view(-1, 1)
        indices = torch.arange(b, device=tgt_seq.device).view(-1, 1).expand(-1, l_tgt).reshape(-1, 1)

        tmp = torch.cat([indices, attn_dic], 1)
        output, inverse = torch.unique(tmp, dim=0, return_inverse=True)

        # anz < b*l_tgt
        dic_rel = dic[:,:-1][output[:, 1]]  # anz x l_dic
        tgt_seq_rel = tgt_seq[output[:, 0]]  # anz x l_tgt

        mesh = tgt_seq_rel.unsqueeze(-2).expand(-1, l_dic, -1).eq(dic_rel.unsqueeze(-1))  # anz x l_dic x l_tgt

        mesh[:, :-1][dic_rel[:, 1:].eq(0).unsqueeze(-1).expand(-1, -1, l_tgt)] = 0  # has to be used because of padding
        #mesh[:, -1] = 0

        # copy in the right way
        mesh[:, 1:, 0] = mesh[:, :-1, -1].clone()

        # remove last row and reshape
        mesh = torch.cat([mesh[:, :, :-1].reshape(mesh.shape[0], -1),
                          torch.zeros(mesh.shape[0], l_dic, dtype=mesh.dtype, device=tgt_seq.device)],
                         1).view(-1, l_dic, l_tgt)  # anz x l_dic x l_tgt

        # cumsum
        mesh = torch.cumsum(mesh, 1)
        mesh[mesh.ne(torch.arange(l_dic, device=tgt_seq.device).view(1, -1, 1) + 1)] = 0

        # add last row and reshape
        mesh = torch.cat([mesh, torch.zeros(mesh.shape[0], mesh.shape[1], 1, dtype=torch.long, device=tgt_seq.device)],2)
        mesh = mesh.view(mesh.shape[0], -1)[:, :l_dic * l_tgt].reshape(mesh.shape[0], l_dic, l_tgt).max(1)[0]  # anz x l_tgt

        mesh = dic[output[:, 1]].gather(1, mesh)

        inverse2 = torch.arange(l_tgt, device=tgt_seq.device).view(1, -1).expand(b, -1).reshape(-1) + l_tgt * inverse
        tgt_ids_out = mesh.view(-1)[inverse2].view(b, l_tgt)  # b x l_tgt
        # tgt_ids_out = mesh[inverse].view(b, l_tgt, l_tgt)[:, torch.arange(l_tgt), torch.arange(l_tgt)] # b x l_tgt

        #print(tgt_ids_out[0],tgt_ids_out.shape)
        #sys.exit()

        return tgt_ids_out

class DecoderLayerMemory(DecoderLayer):
    def __init__(self, d_model, d_inner, n_head, dropout=0.1, n_enc_head=0, rel_pos=False,
                 size_memory=200, version_gate=0):
        super().__init__(d_model, d_inner, n_head, dropout=dropout, n_enc_head=n_enc_head, rel_pos=rel_pos)

        self.attentionMemory = AttentionMemory(d_model, size_memory)
        self.attentionMemoryEntry = AttentionMemoryEntry(n_head, d_model, d_inner, dropout, version_gate)

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