# Deep RNN class
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from .attn import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers, bidirectional=True, dropout=0.2, use_cnn=False, freq_kn=3, freq_std=2):
        super().__init__()

        if use_cnn:
            cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)),
                   nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std))]
            self.cnn = nn.Sequential(*cnn)
            input_size = ((((input_size - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1)*32
        else:
            self.cnn = None

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers,
                        bidirectional=bidirectional, bias=False, dropout=dropout, batch_first=True)

    def forward(self, x, mask=None):
        if self.cnn is not None:
            x = self.cnn(x.unsqueeze(1))
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(x.size(0), x.size(1), -1)
            if mask is not None: mask = mask[:, 0:x.size(1)*4:4]

        if mask is not None:
            lengths = mask.sum(-1)
            x = pack_padded_sequence(x, lengths, batch_first=True)            
            x, _ = self.rnn(x)
            x, _ = pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.rnn(x)

        hidden_size = x.size(2) // 2
        x = x[:, :, :hidden_size] + x[:, :, hidden_size:]
        
        return x, mask

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, layers, n_head=8, shared_emb=True, dropout=0.2, emb_drop=0.):
        super().__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        # Define layers
        self.emb = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.scale = hidden_size**0.5 if shared_emb else 1.
        self.emb_drop = nn.Dropout(emb_drop)
        
        d_k = hidden_size // n_head
        self.attn = MultiHeadAttention(n_head, hidden_size, d_k, dropout=dropout)

        dropout = (0 if layers == 1 else dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, layers, dropout=dropout, batch_first=True)

        self.project = nn.Linear(hidden_size, output_size, bias=True)
        if shared_emb: self.emb.weight = self.project.weight

    def forward(self, inputs, enc_out, attn_mask):
        dec_emb = self.emb(inputs) * self.scale
        dec_emb = self.emb_drop(dec_emb)

        lengths = inputs.gt(0).sum(-1)
        dec_in = pack_padded_sequence(dec_emb, lengths, batch_first=True, enforce_sorted=False)
        dec_out = self.lstm(dec_in)[0]
        dec_out = pad_packed_sequence(dec_out, batch_first=True)[0]

        lt = inputs.size(1)
        attn_mask = attn_mask.eq(0).unsqueeze(1).expand(-1, lt, -1)
        
        context = self.attn(dec_out, enc_out, mask=attn_mask)
        output = context + dec_out
        output = self.project(output)

        return output

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_enc=5, n_dec=2,
                       n_head=8, use_cnn=False, freq_kn=3, freq_std=2, shared_emb=True, dropout=0.2, emb_drop=0.):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(input_size, hidden_size, n_enc, dropout=dropout,
                            use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std)
        self.decoder = Decoder(output_size, hidden_size, n_dec, n_head,
                            shared_emb=shared_emb, dropout=dropout, emb_drop=emb_drop)

    def forward(self, inputs, masks, targets, sampling=0.):
        enc_out, masks = self.encoder(inputs, masks)

        seq_logit = self.decoder(targets, enc_out, masks)
        return seq_logit.view(-1, seq_logit.size(2))

    def encode(self, inputs, masks):
        return self.encoder(inputs, masks)

    def decode(self, enc_out, masks, targets):
        seq_logit = self.decoder(targets, enc_out, masks)
        seq_logit = seq_logit[:,-1,:].squeeze(1)
        return torch.log_softmax(seq_logit, -1)
