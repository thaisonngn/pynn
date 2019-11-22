#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import copy
import argparse

import torch
import torch.nn as nn

from pynn.net.lm import SeqLM

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-layer', type=int, default=2)
parser.add_argument('--d-model', type=int, default=320)
parser.add_argument('--d-project', type=int, default=0)
parser.add_argument('--shared-emb', help='sharing decoder embedding', action='store_true')

parser.add_argument('--states', help='model instance', required=True)
parser.add_argument('--model-path', help='model saving path', default='model')
parser.add_argument('--save-all', help='save configuration as well', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
        
    m_params = {'output_size': args.n_classes,
        'hidden_size': args.d_model,
        'bn_size': args.d_project,
        'layers': args.n_layer,
        'shared_emb': args.shared_emb}
    model = SeqLM(**m_params)

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
        dic = {'params': m_params, 'state': model.state_dict(), 'type': 'lm'}
        torch.save(dic, '%s/epoch-avg.dic' % args.model_path)
    

