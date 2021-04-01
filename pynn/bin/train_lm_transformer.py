#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pynn.util import save_object_param
from pynn.net.lm_transformer import TransformerLM
from pynn.bin import print_model, train_language_model

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--train-seq', help='path to train seq', required=True)
parser.add_argument('--valid-seq', help='path to validation seq', required=True)

parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-layer', type=int, default=4)
parser.add_argument('--d-model', type=int, default=512)
parser.add_argument('--d-inner', type=int, default=0)
parser.add_argument('--d-emb', type=int, default=0)
parser.add_argument('--d-project', type=int, default=0)
parser.add_argument('--n-head', type=int, default=8)
parser.add_argument('--rel-pos', help='relative position', action='store_true')
parser.add_argument('--shared-emb', help='sharing embedding', action='store_true')
parser.add_argument('--no-sek', help='without start and end tokens', action='store_true')

parser.add_argument('--label-smooth', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--emb-drop', type=float, default=0.0)
parser.add_argument('--layer-drop', type=float, default=0.0)
parser.add_argument('--model-path', help='model saving path', default='model')

parser.add_argument('--n-epoch', type=int, default=50)
parser.add_argument('--n-save', type=int, default=5)
parser.add_argument('--n-warmup', help='warm-up steps', type=int, default=6000)
parser.add_argument('--n-const', help='constant steps', type=int, default=0)
parser.add_argument('--n-print', help='inputs per update', type=int, default=40000)
parser.add_argument('--b-input', help='inputs per load', type=int, default=0)
parser.add_argument('--b-sample', help='maximum samples per batch', type=int, default=64)
parser.add_argument('--b-update', help='characters per update', type=int, default=12000)
parser.add_argument('--b-sync', help='steps per update', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--grad-norm', help='divide gradient by updated tokens', action='store_true')
parser.add_argument('--fp16', help='fp16 or not', action='store_true')

def create_model(args, device):
    params = {
        'n_vocab': args.n_classes,
        'd_model': args.d_model,
        'n_layer': args.n_layer,
        'd_inner': args.d_inner,
        'd_emb': args.d_emb,
        'd_project': args.d_project,
        'n_head': args.n_head,
        'shared_emb': args.shared_emb,
        'rel_pos': args.rel_pos,
        'dropout': args.dropout,
        'emb_drop': args.emb_drop,
        'layer_drop': args.layer_drop}
    model = TransformerLM(**params)
    save_object_param(model, params, args.model_path+'/model.cfg')
    return model

def train(device, args):
    model = create_model(args, device)
    print_model(model)
    train_language_model(model, args, device)

def train_distributed(device, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=device, world_size=gpus)
    torch.manual_seed(0)

    model = create_model(args, device)
    if device == 0: print_model(model)
    train_language_model(model, args, device, gpus)

    dist.destroy_process_group()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    if torch.cuda.device_count() > 1:
        gpus = torch.cuda.device_count()
        print('Training with distributed data parallel. Number of devices: %d' % gpus)
        mp.spawn(train_distributed, nprocs=gpus, args=(gpus, args), join=True)
    else:
        device = 0 if torch.cuda.is_available() else torch.device('cpu')
        train(device, args)
