#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pynn.util import save_object_param, load_object
from pynn.net.s2s_transformer_memory import TransformerMemory
from pynn.bin import print_model, train_s2s_model_memory

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--train-scp', help='path to train scp', required=True)
parser.add_argument('--train-target', help='path to train target', required=True)
parser.add_argument('--valid-scp', help='path to validation scp', required=True)
parser.add_argument('--valid-target', help='path to validation target', required=True)

parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-head', type=int, default=8)
parser.add_argument('--n-enc-head', type=int, default=0)
parser.add_argument('--n-enc', type=int, default=4)
parser.add_argument('--n-dec', type=int, default=4)
parser.add_argument('--d-input', type=int, default=80)
parser.add_argument('--d-model', type=int, default=512)
parser.add_argument('--d-inner', type=int, default=1024)
parser.add_argument('--shared-emb', help='shared embedding', action='store_true')
parser.add_argument('--rel-pos', help='relative positional', action='store_true')

parser.add_argument('--time-ds', help='downsample in time axis', type=int, default=1)
parser.add_argument('--use-cnn', help='use CNN filters', action='store_true')
parser.add_argument('--freq-kn', help='frequency kernel', type=int, default=3)
parser.add_argument('--freq-std', help='frequency stride', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--emb-drop', type=float, default=0.0)
parser.add_argument('--enc-drop', type=float, default=0.0)
parser.add_argument('--dec-drop', type=float, default=0.0)
parser.add_argument('--label-smooth', type=float, default=0.1)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--weight-noise', action='store_true')
parser.add_argument('--teacher-force', type=float, default=1.0)

parser.add_argument('--downsample', help='concated frames', type=int, default=4)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--spec-drop', help='argument inputs', action='store_true')
parser.add_argument('--spec-bar', help='number of bars', type=int, default=2)
parser.add_argument('--spec-ratio', help='spec-drop ratio', type=float, default=0.4)
parser.add_argument('--time-stretch', help='argument inputs', action='store_true')
parser.add_argument('--time-win', help='time stretch window', type=int, default=10000)

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

parser.add_argument('--pretrained-model', type=str, default="/export/data1/chuber/memory/tf-24x8-bpe4k/epoch-avg.dic")
parser.add_argument('--bpe-model', type=str, default="/project/asr_systems/LT2021/EN/bpe/m.model")
parser.add_argument('--n-memory', type=int, default=200)
parser.add_argument('--n-enc-mem', type=int, default=8)
parser.add_argument('--n-dec-mem', type=int, default=4)
parser.add_argument('--n-seq-max-epoch', type=int, default=500000)
parser.add_argument('--encode-values', action='store_true')
parser.add_argument('--no-skip-conn', action='store_true')
parser.add_argument('--version-gate', type=int, default=0)
parser.add_argument('--prob-perm', type=float, default=0.5)
parser.add_argument('--clas-model', action='store_true')
parser.add_argument('--freeze-baseline', action='store_true')

def create_model(args, device):
    n_enc_head = args.n_head if args.n_enc_head==0 else args.n_enc_head
    params = {
        'n_vocab': args.n_classes,
        'd_input': args.d_input,
        'd_model': args.d_model,
        'd_inner': args.d_inner,
        'n_enc': args.n_enc,
        'n_enc_head': n_enc_head,
        'n_dec': args.n_dec,
        'n_dec_head': args.n_head,
        'shared_emb': args.shared_emb,
        'rel_pos': args.rel_pos,        
        'time_ds': args.time_ds,
        'use_cnn': args.use_cnn,
        'freq_kn': args.freq_kn,
        'freq_std': args.freq_std,
        'dropout': args.dropout,
        'emb_drop': args.emb_drop,
        'enc_drop': args.enc_drop,
        'dec_drop': args.dec_drop,
        'size_memory': args.n_memory,
        'n_enc_mem': args.n_enc_mem,
        'n_dec_mem': args.n_dec_mem,
        'encode_values': args.encode_values,
        'no_skip_conn_mem': args.no_skip_conn,
        'version_gate': args.version_gate,
        'prob_perm': args.prob_perm,
        'clas_model': args.clas_model}
    model = TransformerMemory(**params)
    save_object_param(model, params, args.model_path+'/model.cfg')
    return model

def train(device, args):
    model = create_model(args, device)
    print_model(model)
    train_s2s_model_memory(model, args, device)

def train_distributed(device, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=device, world_size=gpus)
    torch.manual_seed(0)

    model = create_model(args, device)
    if device == 0:
        print_model(model)
    train_s2s_model_memory(model, args, device, gpus)

    dist.destroy_process_group()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    if torch.cuda.device_count() > 1:
        gpus = torch.cuda.device_count()
        print('Training with distributed data parallel. Number of devices = %d' % gpus)
        mp.spawn(train_distributed, nprocs=gpus, args=(gpus, args), join=True)
    else:
        device = 0 if torch.cuda.is_available() else torch.device('cpu')
        train(device, args)