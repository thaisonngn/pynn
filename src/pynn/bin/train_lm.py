#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from pynn.io.text_seq import TextSeqReader
from pynn.net.lm import SeqLM
from pynn.trainer.adam_lm import train_model

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--train-seq', help='path to train seq', required=True)
parser.add_argument('--valid-seq', help='path to validation seq', required=True)

parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-layer', type=int, default=2)
parser.add_argument('--d-model', type=int, default=320)
parser.add_argument('--d-project', type=int, default=0)
parser.add_argument('--shared-emb', help='sharing decoder embedding', action='store_true')

parser.add_argument('--smooth', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--emb-drop', type=float, default=0.0)
parser.add_argument('--model-path', help='model saving path', default='model')

parser.add_argument('--n-epoch', type=int, default=50)
parser.add_argument('--n-warmup', help='warm-up steps', type=int, default=6000)
parser.add_argument('--n-const', help='constant steps', type=int, default=0)
parser.add_argument('--n-print', help='inputs per update', type=int, default=40000)
parser.add_argument('--b-input', help='inputs per load', type=int, default=32)
parser.add_argument('--b-update', help='characters per update', type=int, default=12000)
parser.add_argument('--shuffle', help='shuffle samples every epoch', action='store_true')
parser.add_argument('--lr', help='learning rate', type=float, default=2.0)

parser.add_argument('--loss-norm', help='per-token loss normalization', action='store_true')
parser.add_argument('--grad-norm', help='per-token grad normalization', action='store_true')
parser.add_argument('--fp16', help='fp16 or not', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    model = SeqLM(
        output_size=args.n_classes,
        hidden_size=args.d_model,
        bn_size=args.d_project,
        layers=args.n_layer,
        shared_emb=args.shared_emb,
        dropout=args.dropout,
        emb_drop=args.emb_drop).to(device)

    tr_reader = TextSeqReader(args.train_seq, fp16=args.fp16, shuffle=args.shuffle)
    cv_reader = TextSeqReader(args.valid_seq, fp16=args.fp16)

    cfg = {'model_path': args.model_path, 'lr': args.lr, 'smooth': args.smooth,
           'n_warmup': args.n_warmup, 'n_const': args.n_const,
           'b_input': args.b_input, 'b_update': args.b_update, 'n_print': args.n_print}
    datasets = (tr_reader, cv_reader)
    train_model(model, datasets, args.n_epoch, device, cfg,
                loss_norm=args.loss_norm, grad_norm=args.grad_norm, fp16=args.fp16)

