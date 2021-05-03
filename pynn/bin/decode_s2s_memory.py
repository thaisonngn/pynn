#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import argparse
import sentencepiece as spm

import torch
import torch.nn.functional as F

from pynn.util import load_object
from pynn.decoder.s2s import beam_search_memory
from pynn.util.text import load_dict, write_hypo
from pynn.io.audio_seq import SpectroDataset

from pynn.io.memory_dataset import MemoryDataset

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

parser.add_argument('--valid-scp', help='path to validation scp', default="/export/data1/chuber/data/sys-All/01-Feats/data_cv.scp")
parser.add_argument('--valid-target', help='path to validation target', default="/export/data1/chuber/data/sys-All/02-Labels/data.bpe4k")

parser.add_argument('--preload', help='preloading ark matrix into memory', action='store_true')
parser.add_argument('--bpe-model', type=str, default="/export/data1/chuber/data/bpe_model/m.model")
parser.add_argument('--n-memory', type=int, default=200)
parser.add_argument('--n-classes', type=int, default=4003)
parser.add_argument('--lstm-based-baseline', default=None)

def printTgt(tgt, sp):
    for i, g in enumerate(tgt[:, 1:]):
        ids = [x.item()%4003 - 2 for x in g]
        ids = [i for i in ids if not i == -2]
        text = sp.decode_ids(ids)
        print(text)

def printStats(tgt_seq, tgt_ids_mem, sp, stats, id, gold, forward, id2):
    stats2 = []
    for j, st in enumerate(stats):
        gate = F.softmax(st, -1)[:,:,0][id]
        samples = st.argmax(-1)[id]-1
        stats2.append((gate, samples))

    maxGold = max([len(g) for g in gold])
    maxForward = max([len(f) for f in forward])

    print("Tgt:", end=" ")
    printTgt(tgt_seq[0:1], sp)

    ids = set()
    ids.add(id2)
    for i in range(len(gold)):
        for j in range(len(stats2)):
            if stats2[j][1][i]!=-1:
                ids.add(int(stats2[j][1][i]))

    for i in range(len(tgt_ids_mem)):
        if i in ids:
            print("%2d" % i + ", Tgt_st", end=": ")
            printTgt(tgt_ids_mem[i:i + 1], sp)

    for i, (g, f) in enumerate(zip(gold, forward)):
        print(("gold: %" + str(maxGold) + "s, forward: %" + str(maxForward) + "s,") % (g, f), end=" ")
        for j in range(len(stats2)):
            print("gate: %.2f, s_id: %3d," % (stats2[j][0][i], stats2[j][1][i]), end=" ")
        print("")

def replaceEntryInStorage(tgt_seq_st, id, label):
    if len(label) > tgt_seq_st.shape[1]:
        z = torch.zeros(tgt_seq_st.shape[0], len(label) - tgt_seq_st.shape[1], device=tgt_seq_st.device,
                        dtype=tgt_seq_st.dtype)
        tgt_seq_st = torch.cat([tgt_seq_st, z], 1)

    for i in range(tgt_seq_st.shape[1]):
        if i < len(label):
            tgt_seq_st[id, i] = label[i]
        else:
            tgt_seq_st[id, i] = 0

    return tgt_seq_st

def encode(self, src_seq, src_mask, tgt_ids_mem):
    enc_out, enc_mask = self.encoder(src_seq, src_mask)
    enc_out2, enc_mask2 = self.encoder2(src_seq, src_mask)[:2]

    # generate tgt and gold sequence in the memory
    tgt_emb_mem = self.decoder_mem.emb(tgt_ids_mem)
    tgt_mask_mem = tgt_ids_mem.ne(0)

    if self.decoder_mem.rel_pos:
        raise NotImplementedError
        # pos_emb = self.pos.embed(tgt_emb_mem)
    else:
        pos_seq = torch.arange(0, tgt_emb_mem.size(1), device=tgt_emb_mem.device, dtype=tgt_emb_mem.dtype)
        pos_emb = self.decoder_mem.pos(pos_seq, tgt_emb_mem.size(0))
        tgt_emb_mem = tgt_emb_mem * self.decoder_mem.scale + pos_emb
    tgt_seq_mem = self.decoder_mem.emb_drop(tgt_emb_mem)

    # encode tgt seq from the memory
    enc_out_mem = self.encoder_mem(tgt_seq_mem, tgt_mask_mem)[0]

    # calc mean
    enc_out_mem[tgt_mask_mem.logical_not()] = 0
    enc_out_mem_mean = enc_out_mem.sum(1) / (tgt_mask_mem.sum(1, keepdims=True))  # n_mem x d_model

    enc_out_mem_mean = torch.cat([self.no_entry_found, enc_out_mem_mean], 0)

    if not self.encode_values:
        return enc_out, enc_mask, tgt_emb_mem, tgt_mask_mem, enc_out_mem, enc_out_mem_mean, enc_out2, enc_mask2
    else:
        return enc_out, enc_mask, enc_out_mem, tgt_mask_mem, enc_out_mem, enc_out_mem_mean, enc_out2, enc_mask2

def decode(self, tgt_seq, enc_out, label_mem=None, gold=None, inference=True):
    enc_out, enc_mask, tgt_emb_mem, tgt_mask_mem, enc_out_mem, enc_out_mem_mean, enc_out2, enc_mask2 = enc_out

    dec_out_orig = self.decoder2(tgt_seq, enc_out2, enc_mask2)[0]
    dec_out_mem, mem_attn_outs = self.decoder_mem(tgt_seq, enc_out, enc_mask, tgt_emb_mem, tgt_mask_mem, enc_out_mem,
                                                  enc_out_mem_mean)

    dec_out_orig = F.softmax(dec_out_orig.to(torch.float32), -1)
    dec_out_mem = F.softmax(dec_out_mem.to(torch.float32), -1)

    if not label_mem is None:
        label_gate = label_mem.clamp(max=1)

        mask = gold.gt(2)

        dec_out_orig = self.noise_permute(dec_out_orig, gold, label_gate.eq(1) & mask)
        dec_out_mem = self.noise_permute(dec_out_mem, gold, label_gate.eq(0) & mask)

    gates = torch.cat([F.softmax(a.to(torch.float32), -1)[:, :, 0:1].to(a.dtype).detach() for a in mem_attn_outs], -1)
    gates = F.sigmoid(self.project(gates).to(torch.float32))

    dec_output = gates * dec_out_orig + (1 - gates) * dec_out_mem

    if not inference:
        return dec_output, mem_attn_outs
    else:
        dec_output = dec_output[:, -1, :].squeeze(1)
        return torch.log(dec_output)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    dic, word_dic = load_dict(args.dict, args.word_dict)

    use_gpu = torch.cuda.is_available()
    args.device = device = torch.device('cuda' if use_gpu else 'cpu')
    print('Using device: ' + str(device))

    mdic = torch.load(args.model_dic)
    model = load_object(mdic['class'], mdic['module'], mdic['params'])
    model = model.to(device)
    model.load_state_dict(mdic['state'])

    if args.lstm_based_baseline is not None:
        mdic = torch.load(args.lstm_based_baseline)
        model2 = load_object(mdic['class'], mdic['module'], mdic['params'])
        model2 = model2.to(device)
        model2.load_state_dict(mdic['state'])

        model.encoder2 = model2.encoder
        model.decoder2 = model2.decoder

        import types

        model.encode = types.MethodType(encode, model)
        model.decode = types.MethodType(decode, model)

    model.eval()
    if args.fp16: model.half()

    print("Model loaded")

    lm = None
    if args.lm_dic is not None:
        mdic = torch.load(args.lm_dic)
        lm = load_object(mdic['class'], mdic['module'], mdic['params'])
        lm = lm.to(device)
        lm.load_state_dict(mdic['state'])
        lm.eval()
        if args.fp16: lm.half()

    print("Initialize Datasets.....")
    print("Loading ...")
    cv_data = SpectroDataset(args.valid_scp, args.valid_target, downsample=args.downsample,
                             sort_src=False, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             threads=2, verbose=False)
    print("Loading ...")
    cv_data.initialize(b_input=2500, b_sample=64)
    print("Loading ...")
    cv_data = MemoryDataset(cv_data, args, validation=True, fast=True)

    # get storage data
    _, _, _, tgt_ids_mem_save, _ = cv_data[42]
    tgt_ids_mem_save = tgt_ids_mem_save.to(device)

    print("Loading ...")

    # ---------- DECODE NEW WORDS TESTSET --------------

    # makes segmenter instance and loads the model file (.model)
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    # dataset to decode
    data_scp = "/export/data1/chuber/data/sys-tests/new-words-test/test.scp"
    data_label = "/export/data1/chuber/data/sys-tests/new-words-test/test.bpe"
    data_new_words = "/export/data1/chuber/data/sys-tests/new-words-test/data.txt"

    words = []
    for line in open(data_new_words, "r"):
        if line[0] == "-":
            continue
        words.append(" ".join(line.strip().split()[2:]))

    dataset = SpectroDataset(data_scp, data_label, downsample=args.downsample,
                             sort_src=False, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             threads=2, verbose=False)
    dataset.initialize(b_input=2500, b_sample=64)

    id2 = 0
    for id, word in zip(range(len(dataset)), words):
        for changeStorage in [False, True]:
            print("Storage new words:", changeStorage)

            sample_batched = dataset.collate_fn([dataset[id]])
            src_seq, src_mask, tgt_seq = map(lambda x: x.to(device), sample_batched)

            tgt_ids_mem = copy.deepcopy(tgt_ids_mem_save)

            # add part of data to decode to storage
            if changeStorage:
                word2 = cv_data.encode_as_ids(word)
                tgt_ids_mem = replaceEntryInStorage(tgt_ids_mem, id2, word2)

            gold = tgt_seq[:, 1:]
            tgt_seq = tgt_seq[:, :-1]

            # run the model
            with torch.no_grad():
                pred, stats = model(src_seq, src_mask, tgt_seq, tgt_ids_mem)[:2]

            pred = pred.argmax(-1)

            printStats(tgt_seq, tgt_ids_mem, sp, stats, 0, [sp.decode_ids([int(x) - 2]) for x in gold[0]],
                       [sp.decode_ids([int(x) - 2]) for x in pred[0]], id2)

            with torch.no_grad():
                hypos, scores = beam_search_memory(model, src_seq, src_mask, tgt_ids_mem, device, args.beam_size,
                                                   args.max_len, len_norm=args.len_norm, lm=lm, lm_scale=args.lm_scale)

            # print(pred[0])
            print("Output decode:", " ".join([x for x in sp.decode_ids([int(x) - 2 for x in hypos[0] if not x == 2]).split() if not x=="<unk>"]))

    # ------- DECODE TED TEST SET ----------------

    reader = SpectroDataset(args.data_scp, mean_sub=args.mean_sub, fp16=args.fp16,
                            downsample=args.downsample)
    since = time.time()
    fout = open(args.output, 'w')
    with torch.no_grad():
        while True:
            seq, mask, utts = reader.read_batch_utt(args.batch_size)
            if not utts: break
            #print("Decoding shape "+str(seq.shape))
            seq, mask = seq.to(device), mask.to(device)
            hypos, scores = beam_search_memory(model, seq, mask, copy.deepcopy(tgt_ids_mem_save), device, args.beam_size,
                                               args.max_len, len_norm=args.len_norm, lm=lm, lm_scale=args.lm_scale)
            hypos, scores = hypos.tolist(), scores.tolist()
            write_hypo(hypos, scores, fout, utts, dic, word_dic, args.space, args.format)
    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
