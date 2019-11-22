#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from pynn.io.kaldi_seq import KaldiStreamLoader
from pynn.net.rnn_ctc import DeepLSTM
from pynn.trainer.adam_ctc import train_model

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--train-scp', help='path to train scp', required=True)
parser.add_argument('--train-target', help='path to train target', required=True)
parser.add_argument('--valid-scp', help='path to validation scp', required=True)
parser.add_argument('--valid-target', help='path to validation target', required=True)

parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-layers', type=int, default=4)
parser.add_argument('--d-hidden', type=int, default=320)
parser.add_argument('--d-input', type=int, default=40)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--channels', help='cnn channels', type=int, default=1)
parser.add_argument('--uni-direct', help='uni directional', action='store_true')
parser.add_argument('--smooth', type=float, default=0.1)
parser.add_argument('--augment', help='argument inputs', action='store_true')
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--max-len', help='max sequence length', type=int, default=5000)
parser.add_argument('--model-path', help='model saving path', default='model')

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--n-warmup', help='warm-up steps', type=int, default=8000)
parser.add_argument('--n-print', help='inputs per update', type=int, default=5000)
parser.add_argument('--b-input', help='inputs per batch', type=int, default=3000)
parser.add_argument('--b-update', help='characters per update', type=int, default=12000)
parser.add_argument('--lr', help='learning rate', type=float, default=2.0)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    bidirect = not args.uni_direct
    model = DeepLSTM(
        input_size=args.d_input,
        hidden_size=args.d_hidden,
        layers=args.n_layers,
        n_classes=args.n_classes,
        channels=args.channels,
        bidirectional=bidirect,
        dropout=args.dropout).to(device)
        
    tr_loader = KaldiStreamLoader(args.train_scp, args.train_target, downsample=args.downsample,
                                  sort_src=True, max_len=args.max_len, augment=args.augment, sek=False)
    cv_loader = KaldiStreamLoader(args.valid_scp, args.valid_target, downsample=args.downsample,
                                  sort_src=True, max_len=args.max_len, sek=False)
    
    cfg = {'model_path': args.model_path, 'lr': args.lr, 'smooth': args.smooth,
           'n_warmup': args.n_warmup, 'b_input': args.b_input, 'b_update': args.b_update,
           'n_print': args.n_print}
    datasets = (tr_loader, cv_loader)
    train_model(model, datasets, args.epochs, device, cfg)

