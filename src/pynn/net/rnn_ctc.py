# Deep RNN class

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers, n_classes, bidirectional=True, dropout=0., channels=0):
        super(DeepLSTM, self).__init__()

        if channels > 0:
            cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2),
                   nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)]
            self.cnn = nn.Sequential(*cnn)
            input_size = (((input_size // channels) - 3) // 4)*32
        else:
            self.cnn = None
        self.channels = channels

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers,
                        bidirectional=bidirectional, bias=False, dropout=dropout, batch_first=True)

        self.fc = nn.Linear(hidden_size, n_classes, bias=True)

        # Initilziation
        self.init_fotget_gate()
        self.init_weight()

    def forward(self, x, mask=None):
        if isinstance(x, PackedSequence):
            x, _ = self.rnn(x)
            x, _ = pad_packed_sequence(x, batch_first=True)
        else:
            if self.cnn is not None:
                feqs = x.size(2) // self.channels
                x = x.view(x.size(0), x.size(1), feqs, self.channels)
                x = x.permute(0, 3, 1, 2)
                x = self.cnn(x)
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
        x = self.fc(x)
 
        return x, mask

    def init_fotget_gate(self):
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

    def init_weight(self):
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "weight" in n, names):
                w = getattr(self.rnn, name)
                r = math.sqrt(1. / w.data.size(1))
                w.data.uniform_(-0.1, 0.1)
