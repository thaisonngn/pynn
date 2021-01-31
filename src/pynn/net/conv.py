# Copyright 2020 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SubsamplingConv(nn.Module):
    def __init__(self, freq_kn=3, freq_std=2):
        super().__init__()

        cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)),
               nn.ReLU(),
               nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)),
               nn.ReLU()]
        self.cnn = nn.Sequential(*cnn)
        d_input = ((((d_input - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1)*32


    def forward(self, seq, mask):
        seq = seq.unsqueeze(1)
        seq = self.cnn(seq)
        seq = seq.permute(0, 2, 1, 3).contiguous()
        seq = seq.view(seq.size(0), seq.size(1), -1)
        if mask is not None: mask = mask[:, 0:seq.size(1)*4:4]

        return seq, mask

