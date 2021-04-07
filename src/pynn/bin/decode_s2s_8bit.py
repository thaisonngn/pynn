#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import argparse
import os

import torch

from pynn.util import load_object
from pynn.decoder.s2s import beam_search, beam_search_cache
from pynn.util.text import load_dict, write_hypo
from pynn.io.audio_seq import SpectroDataset

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--lm-dic', help='language model dictionary', default=None)
parser.add_argument('--lm-scale', help='language model scale', type=float, default=0.5)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--word-dict', help='word dictionary file', default=None)
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--state-cache', help='caching encoder states', action='store_true')
parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--beam-size', help='beam size', type=int, default=10)
parser.add_argument('--max-len', help='max len', type=int, default=100)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--len-norm', help='length normalization', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='hypos/H_1_LV.ctm')
parser.add_argument('--format', help='output format', type=str, default='ctm')
parser.add_argument('--space', help='space token', type=str, default='▁')

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    
if __name__ == '__main__':
    args = parser.parse_args()

    dic, word_dic = load_dict(args.dict, args.word_dict)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.model_dic)
    state = {}
    for name, param in mdic['state'].items():
        if name.endswith('_raw'):
            name = name[:-len('_raw')]
        state[name] = param

    module = "pynn.net.s2s_lstm_8bit"
    model = load_object(mdic['class'], module, mdic['params'])
    #model = model.to(device)
    #model.load_state_dict(mdic['state'])
    print(module)
    model.load_state_dict(state, strict=False)
    mdic = {'params': mdic['params'], 'class': mdic['class'], 'module': module, 'state': model.state_dict()}
    torch.save(mdic, 'epoch-avg.dic')
    model.eval()

    print_size_of_model(model)
    model = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.LSTM, torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights
    print_size_of_model(model)
    print(model)

    model = model.to(device)
    
    lm = None
    if args.lm_dic is not None:
        mdic = torch.load(args.lm_dic)
        lm = load_object(mdic['class'], mdic['module'], mdic['params'])
        lm = lm.to(device)
        lm.load_state_dict(mdic['state'])
        lm.eval()
        if args.fp16: lm.half()
        
    reader = SpectroDataset(args.data_scp, mean_sub=args.mean_sub, fp16=args.fp16,
                            downsample=args.downsample)
    since = time.time()
    fout = open(args.output, 'w')
    decode_func = beam_search_cache if args.state_cache else beam_search
    with torch.no_grad():
        while True:
            seq, mask, utts = reader.read_batch_utt(args.batch_size)
            if not utts: break
            seq, mask = seq.to(device), mask.to(device)
            hypos, scores = decode_func(model, seq, mask, device, args.beam_size,
                                        args.max_len, len_norm=args.len_norm, lm=lm, lm_scale=args.lm_scale)
            hypos, scores = hypos.tolist(), scores.tolist()
            write_hypo(hypos, scores, fout, utts, dic, word_dic, args.space, args.format)
    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
