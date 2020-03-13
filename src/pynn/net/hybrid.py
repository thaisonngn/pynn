# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tf import Encoder
from .seq2seq import Decoder

class Hybrid(nn.Module):
    def __init__(self, n_vocab, d_input, d_model, n_enc=8, n_dec=2,
            n_head=8, dropout=0.2, emb_drop=0., enc_drop=0.,
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2, lm=True, shared_emb=True):

        super(Hybrid, self).__init__()

        self.encoder = Encoder(d_input=d_input, d_model=d_model, d_inner=d_model*2,
                            n_layers=n_enc, n_head=8, d_k=d_model//8,
                            time_ds=time_ds, use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std,
                            dropout=dropout, layer_drop=enc_drop)

        self.decoder = Decoder(n_vocab, d_model, n_dec, n_head=n_head,
                            lm=lm, shared_emb=shared_emb, dropout=dropout, emb_drop=emb_drop)

    def forward(self, inputs, masks, targets):
        enc_out, masks = self.encoder(inputs, masks)
        logit = self.decoder(targets, enc_out, masks)[0]
        #return logit.view(-1, logit.size(2))
        return logit

    def encode(self, inputs, masks):
        return self.encoder(inputs, masks)

    def decode(self, enc_out, masks, targets):
        logit = self.decoder(targets, enc_out, masks)[0]
        logit = logit[:,-1,:].squeeze(1)
        return torch.log_softmax(logit, -1)
