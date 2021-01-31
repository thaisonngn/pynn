#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os, glob
import copy
import argparse

import torch

from pynn.util import load_object_param

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-path', help='model saving path', default='model')
parser.add_argument('--config', help='model config', default='model.cfg')
parser.add_argument('--states', help='model states', default='ALL')
parser.add_argument('--save-all', help='save configuration as well', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()

    model, cls, module, m_params = load_object_param(args.model_path + '/' + args.config)

    ext = copy.deepcopy(model)
    if args.states == 'ALL':
        states = [s for s in glob.glob("%s/epoch-*.pt" % args.model_path)]
    else:
        states = args.states.split(',')
        states = ["%s/epoch-%s.pt" % (args.model_path, s) for s in states]
        
    state = states[0]
    model.load_state_dict(torch.load(state, map_location='cpu'))
    params = list(model.parameters())
    for state in states[1:]:
        ext.load_state_dict(torch.load(state, map_location='cpu'))
        eparams = list(ext.parameters())
        for i in range(len(params)):
            params[i].data.add_(eparams[i].data)
    scale = 1.0 / len(states)
    for p in params: p.data.mul_(scale)
   
    state = model.state_dict() 
    if not args.save_all:
        model_file = '%s/epoch-avg.pt' % args.model_path
        torch.save(state, model_file)
    else:
        dic = {'params': m_params, 'class': cls, 'module': module, 'state': state}
        torch.save(dic, '%s/epoch-avg.dic' % args.model_path)
