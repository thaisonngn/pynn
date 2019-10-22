# Licensed under the Apache License, Version 2.0 (the "License");

import os
import copy
import argparse

import torch
import torch.nn as nn

from pynn.net.seq2seq import Seq2Seq

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-head', type=int, default=8)
parser.add_argument('--n-enc', type=int, default=4)
parser.add_argument('--n-dec', type=int, default=2)
parser.add_argument('--d-input', type=int, default=40)
parser.add_argument('--d-model', type=int, default=320)

parser.add_argument('--time-ds', help='downsample in time axis', type=int, default=1)
parser.add_argument('--use-cnn', help='use CNN filters', action='store_true')
parser.add_argument('--freq-kn', help='frequency kernel', type=int, default=3)
parser.add_argument('--freq-std', help='frequency stride', type=int, default=2)
parser.add_argument('--shared-emb', help='sharing decoder embedding', action='store_true')

parser.add_argument('--states', help='model instance', required=True)
parser.add_argument('--model-path', help='model saving path', default='model')
parser.add_argument('--save-all', help='save configuration as well', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()

    m_params = {'input_size': args.d_input,
        'hidden_size': args.d_model,
        'output_size': args.n_classes,
        'n_enc': args.n_enc,
        'n_dec': args.n_dec,
        'n_head': args.n_head,
        'time_ds': args.time_ds,
        'use_cnn': args.use_cnn,
        'freq_kn': args.freq_kn,
        'freq_std': args.freq_std,
        'shared_emb': args.shared_emb}
    model = Seq2Seq(**m_params)

    ext = copy.deepcopy(model)
    states = args.states.split(',')
        
    state = "%s/epoch-%s.pt" % (args.model_path, states[0])
    model.load_state_dict(torch.load(state, map_location='cpu'))
    params = list(model.parameters())
    for s in states[1:]:
        state = "%s/epoch-%s.pt" % (args.model_path, s)
        ext.load_state_dict(torch.load(state, map_location='cpu'))
        eparams = list(ext.parameters())
        for i in range(len(params)):
            params[i].data.add_(eparams[i].data)
    scale = 1.0 / len(states)
    for p in params: p.data.mul_(scale)
    
    if not args.save_all:
        model_file = '%s/epoch-avg.pt' % args.model_path
        torch.save(model.state_dict(), model_file)
    else:
        dic = {'params': m_params, 'state': model.state_dict(), 'type': 'lstm'}
        torch.save(dic, '%s/epoch-avg.dic' % args.model_path)
    

