#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

import torch

from pynn.util import load_object, save_object_param
from pynn.io.audio_seq import SpectroDataset
from pynn.net.aligner import AttnAligner
from pynn.trainer.adam_aligner import train_model

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--train-scp', help='path to train scp', required=True)
parser.add_argument('--train-target', help='path to train target', required=True)
parser.add_argument('--valid-scp', help='path to validation scp', required=True)
parser.add_argument('--valid-target', help='path to validation target', required=True)

parser.add_argument('--s2s-model', help='model dictionary', required=True)
parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-layer', type=int, default=4)
parser.add_argument('--d-model', type=int, default=320)
parser.add_argument('--rel-pos', help='relative positional', action='store_true')

parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--layer-drop', type=float, default=0.0)

parser.add_argument('--spec-drop', help='specaugment', action='store_true')
parser.add_argument('--spec-bar', help='number of bars of spec-drop', type=int, default=2)
parser.add_argument('--spec-ratio', help='spec-drop ratio', type=float, default=0.2)
parser.add_argument('--time-stretch', help='argument inputs', action='store_true')
parser.add_argument('--time-win', help='time stretch window', type=int, default=10000)

parser.add_argument('--preload', help='preloading ark matrix into memory', action='store_true')
parser.add_argument('--model-path', help='model saving path', default='model')

parser.add_argument('--n-epoch', type=int, default=100)
parser.add_argument('--n-save', type=int, default=5)
parser.add_argument('--n-warmup', help='warm-up steps', type=int, default=6000)
parser.add_argument('--n-const', help='constant steps', type=int, default=0)
parser.add_argument('--n-print', help='sequences per print', type=int, default=5000)
parser.add_argument('--b-input', help='total input per batch', type=int, default=3000)
parser.add_argument('--b-sample', help='maximum samples per batch', type=int, default=64)
parser.add_argument('--b-update', help='tokens per update', type=int, default=8000)
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--grad-norm', help='divide gradient by updated tokens', action='store_true')
parser.add_argument('--fp16', help='fp16 or not', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.s2s_model)
    s2s = load_object(mdic['class'], mdic['module'], mdic['params'])
    s2s.load_state_dict(mdic['state'])
    s2s.eval()

    params = {
        'n_vocab': args.n_classes,
        'd_model': args.d_model,
        'n_layer': args.n_layer,
        'rel_pos': args.rel_pos,
        'dropout': args.dropout}
    model = AttnAligner(**params)
    save_object_param(model, params, args.model_path+'/model.cfg')

    tr_data = SpectroDataset(args.train_scp, args.train_target, downsample=args.downsample,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             spec_drop=args.spec_drop, spec_bar=args.spec_bar, spec_ratio=args.spec_ratio,
                             time_stretch=args.time_stretch, time_win=args.time_win, threads=2)
    cv_data = SpectroDataset(args.valid_scp, args.valid_target, downsample=args.downsample, threads=2,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload)

    cfg = {'model_path': args.model_path, 'lr': args.lr, 'grad_norm': args.grad_norm,
           'n_warmup': args.n_warmup, 'n_const': args.n_const, 'n_save': args.n_save, 'n_print': args.n_print,
           'b_input': args.b_input, 'b_sample': args.b_sample, 'b_update': args.b_update}
    datasets = (tr_data, cv_data)
    train_model(model, s2s, datasets, args.n_epoch, device, cfg, fp16=args.fp16)

