import time
import os
import copy
import random
import numpy as np
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pynn.util.decoder import Decoder
from pynn.io.kaldi_seq import KaldiStreamLoader
from pynn.net.tf import Transformer
 
parser = argparse.ArgumentParser(description='pynn')

parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-head', type=int, default=8)
parser.add_argument('--n-enc', type=int, default=4)
parser.add_argument('--n-enc-head', type=int, default=0)
parser.add_argument('--n-dec', type=int, default=4)
parser.add_argument('--d-input', type=int, default=80)
parser.add_argument('--d-model', type=int, default=512)
parser.add_argument('--d-inner-hid', type=int, default=1024)
parser.add_argument('--d-k', type=int, default=64)

parser.add_argument('--use-cnn', help='use CNN filters', action='store_true')
parser.add_argument('--shared-kv', help='sharing key and value weights', action='store_true')
parser.add_argument('--shared-emb', help='sharing decoder embedding', action='store_true')
parser.add_argument('--attn-mode', help='encoder attention mode',  type=int, default=0)
parser.add_argument('--model', help='model file', required=True)
parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--word-dict', help='word dictionary file', default=None)
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=4)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--beam-size', help='beam size', type=int, default=10)
parser.add_argument('--max-len', help='max len', type=int, default=400)
parser.add_argument('--output', help='output file', type=str, default='hypos/H_1_LV.ctm')
parser.add_argument('--format', help='output format', type=str, default='ctm')
parser.add_argument('--space', help='space token', type=str, default='<space>')

if __name__ == '__main__':
    args = parser.parse_args()

    dic = None
    if args.dict is not None:
        dic = {}
        fin = open(args.dict, 'r')
        for line in fin:
            tokens = line.split()
            dic[int(tokens[1])] = tokens[0]
    word_dic = None
    if args.word_dict is not None:
        fin = open(args.word_dict, 'r')
        word_dic = {}
        for line in fin:
            tokens = line.split()
            word_dic[''.join(tokens[1:])] = tokens[0]

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    n_enc_head = args.n_head if args.n_enc_head==0 else args.n_enc_head
    m_params = {'n_vocab':args.n_classes,
        'd_input': args.d_input,
        'd_k': args.d_k,
        'd_model': args.d_model,
        'd_inner': args.d_inner_hid,
        'n_enc': args.n_enc,
        'n_enc_head': n_enc_head,
        'n_dec': args.n_dec,
        'n_dec_head': args.n_head,
        'use_cnn': args.use_cnn,
        'shared_kv': args.shared_kv,
        'shared_emb': args.shared_emb,        
        'attn_mode': args.attn_mode}
    model = Transformer(**m_params).to(device)
    
    model.load_state_dict(torch.load(args.model))
    model.eval()

    data_loader = KaldiStreamLoader(args.data_scp, mean_sub=args.mean_sub, downsample=args.downsample)
    data_loader.initialize()

    since = time.time()
    batch_size = args.batch_size
    fout = open(args.output, 'w')
    while True:
        src_seq, src_mask, utts = data_loader.read_batch_utt(batch_size)
        if len(utts) == 0: break
        with torch.no_grad():
            src_seq, src_mask = src_seq.to(device), src_mask.to(device)
            hypos, scores = Decoder.beam_search(model, src_seq, src_mask,
                                            device, args.beam_size, args.max_len)
            hypos, scores = hypos.tolist(), scores.tolist()
            if args.format == 'ctm':
                Decoder.write_to_ctm(hypos, scores, fout, utts, dic, word_dic, args.space)
            else:
                Decoder.write_to_text(hypos, scores, fout, utts, dic, args.space)
    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
