#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

import torch

from pynn.util import save_object_param
from pynn.net.s2s_transformer import Transformer
from pynn.bin import print_model
from pynn.trainer.adam_s2s import train_model 
from pynn.io.text_seq import TextSeqDataset, TextPairDataset

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--train-data', help='path to train scp', required=True)
parser.add_argument('--valid-data', help='path to validation scp', required=True)

parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-emb', type=int, required=True)
parser.add_argument('--n-head', type=int, default=4)
parser.add_argument('--n-enc-head', type=int, default=0)
parser.add_argument('--n-enc', type=int, default=2)
parser.add_argument('--n-dec', type=int, default=2)
parser.add_argument('--d-input', type=int, default=256)
parser.add_argument('--d-model', type=int, default=256)
parser.add_argument('--d-inner', type=int, default=512)
parser.add_argument('--shared-emb', help='shared embedding', action='store_true')
parser.add_argument('--rel-pos', help='relative positional', action='store_true')

parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--emb-drop', type=float, default=0.0)
parser.add_argument('--enc-drop', type=float, default=0.0)
parser.add_argument('--dec-drop', type=float, default=0.0)
parser.add_argument('--label-smooth', type=float, default=0.1)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--weight-noise', action='store_true')
parser.add_argument('--teacher-force', type=float, default=1.0)

parser.add_argument('--preload', help='preloading ark matrix into memory', action='store_true')
parser.add_argument('--model-path', help='model saving path', default='model')

parser.add_argument('--n-epoch', type=int, default=100)
parser.add_argument('--n-save', type=int, default=5)
parser.add_argument('--n-warmup', help='warm-up steps', type=int, default=6000)
parser.add_argument('--n-const', help='constant steps', type=int, default=0)
parser.add_argument('--n-print', help='inputs per update', type=int, default=5000)
parser.add_argument('--b-input', help='inputs per batch', type=int, default=3000)
parser.add_argument('--b-sample', help='maximum samples per batch', type=int, default=64)
parser.add_argument('--b-update', help='characters per update', type=int, default=8000)
parser.add_argument('--b-sync', help='steps per update', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0012)
parser.add_argument('--grad-norm', help='divide gradient by updated tokens', action='store_true')
parser.add_argument('--fp16', help='fp16 or not', action='store_true')

def create_model(args, device):
    n_enc_head = args.n_head if args.n_enc_head==0 else args.n_enc_head
    params = {
        'n_vocab': args.n_classes,
        'n_emb': args.n_emb,
        'd_input': args.d_input,
        'd_model': args.d_model,
        'd_inner': args.d_inner,
        'n_enc': args.n_enc,
        'n_enc_head': n_enc_head,
        'n_dec': args.n_dec,
        'n_dec_head': args.n_head,
        'shared_emb': args.shared_emb,
        'rel_pos': args.rel_pos,        
        'dropout': args.dropout,
        'emb_drop': args.emb_drop,
        'enc_drop': args.enc_drop,
        'dec_drop': args.dec_drop}
    model = Transformer(**params)
    save_object_param(model, params, args.model_path+'/model.cfg')
    return model

def train(device, args):
    model = create_model(args, device)
    print_model(model)

    tr_data = TextPairDataset(args.train_data, threads=2)
    cv_data = TextPairDataset(args.valid_data, threads=2)
    
    cfg = {'model_path': args.model_path, 'lr': args.lr, 'grad_norm': args.grad_norm,
           'weight_decay': args.weight_decay, 'weight_noise': args.weight_noise,
           'n_warmup': args.n_warmup, 'n_const': args.n_const, 'n_save': args.n_save, 'n_print': args.n_print,
           'b_input': args.b_input, 'b_sample': args.b_sample, 'b_update': args.b_update, 'b_sync': args.b_sync}

    datasets = (tr_data, cv_data)
    train_model(model, datasets, args.n_epoch, device, cfg, fp16=args.fp16)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    device = 0 if torch.cuda.is_available() else torch.device('cpu')
    train(device, args)
