#!/usr/bin/python

import time
import subprocess
import os
import io
import re

import numpy
import torch

from pynn.util import audio
from pynn.decoder.s2s import Beam
from pynn.util import load_object

import threading
import sentencepiece as spm

def token2word(tokens, dic, space='', cleaning=True):
    tokens.append(2)
    hypo, pw = [], ''
    for tid in tokens:
        if tid == 2:
            if pw != '': hypo.append(pw)
            break
        token = dic[tid-2]
        if space == '':
            hypo.append(token)
        else:
            if token.startswith(space):
                if pw != '': hypo.append(pw)
                pw = token.replace(space, '') if token != space else ''
            else:
                pw += token
    if cleaning:
        hypo = ['<unk>' if w.startswith('%') or w.startswith('+') or w.startswith('<') or \
                w.startswith('-') or w.endswith('-') else w for w in hypo]
        words, pw = [], ''
        for w in hypo:
            if w == '<unk>' and pw == w: continue
            words.append(w)
            pw = w
        hypo = words

    return hypo

def incl_search(model, src, max_node=8, max_len=10, states=[1], len_norm=False, prune=1.0):
    src_mask = torch.ones((1, src.size(0)), dtype=torch.uint8, device=src.device)
    enc_out = model.encode(src.unsqueeze(0), src_mask)

    beam = Beam(max_node, [1], len_norm)
    if len(states) > 1:
        seq = torch.LongTensor(states).to(src.device).view(1, -1)
        logits = model.get_logit(enc_out, seq)
        logits = logits.squeeze(0)
        for i in range(len(states)-1):
            token = states[i+1]
            prob = logits[i][token]
            beam.advance(0, [prob], [token])
            beam.prune()

    for step in range(max_len):
        l = 1 if step == 0 else max_node
        seq = [beam.seq(k) for k in range(l)]
        seq = torch.LongTensor(seq).to(src.device)

        dec_out = model.decode(enc_out, seq)
        
        probs, tokens = dec_out.topk(max_node, dim=1)
        probs, tokens = probs.cpu().numpy(), tokens.cpu().numpy()

        for k in range(l):
            prob, token = probs[k], tokens[k]
            beam.advance(k, prob, token)

        beam.prune()
        if beam.done: break
    hypo, prob = beam.best_hypo()
    sth = beam.stable_hypo(prune)

    return enc_out, hypo, prob, sth

def init_asr_model(args):
    dic = None
    if args.dict is not None:
        dic = {}
        fin = open(args.dict, 'r')
        for line in fin:
            tokens = line.split()
            dic[int(tokens[1])] = tokens[0]

    device = torch.device(args.device)

    mdic = torch.load(args.model_dic)
    model = load_object(mdic['class'], mdic['module'], mdic['params'])
    model = model.to(device)
    model.load_state_dict(mdic['state'])
    model.eval()

    model = ModelWrapper(model,args)

    return model, device, dic

def decode(model, device, args, adc, fbank_mat, start=0, prefix=[1]):
    signal = numpy.frombuffer(adc[start*16*10*2:], numpy.int16)
    feats = audio.extract_fbank(signal, fbank_mat, sample_rate=16000)
    feats = feats - feats.mean(axis=0, keepdims=True)
     
    frames = (feats.shape[0])
    print("Decoding for audio segment of %d frames" % frames)
    if frames < 10: return [], None, None
    
    space, beam_size, max_len = args.space, args.beam_size, args.max_len
    win, stable_time = args.incl_block, args.stable_time
    head, padding = args.attn_head, args.attn_padding

    with torch.no_grad():
        src = torch.HalfTensor(feats) if args.fp16 else torch.FloatTensor(feats)
        src = src.to(device)
        enc_out, hypo, score, sth = incl_search(model, src, beam_size, max_len, prefix)

        tgt = torch.LongTensor(hypo).to(device).view(1, -1)
        attn = model.get_attn(enc_out, tgt)
        attn = attn[0]
        cs = torch.cumsum(attn[head], dim=1)
        ep = cs.le(1.-padding).sum(dim=1)
        ep = ep.cpu().numpy() * 4
        sp = sp = cs.le(padding).sum(dim=1)
        sp = sp.cpu().numpy() * 4

        best_memory_entry = model.get_best_memory_entry(enc_out, tgt)[0].tolist()

    return hypo, sp, ep, frames, best_memory_entry

class ModelWrapper():
    def __init__(self, model, args):
        self.model = model

        self.tgt_ids_mem = torch.ones(1,2,dtype=torch.int64,device=args.device)
        self.tgt_ids_mem[:,1] = 2

        self.lock = threading.Lock()

        self.sp = spm.SentencePieceProcessor()
        self.sp.load("/".join(args.dict.split("/")[:-1])+"/m.model")

        self.words = None
        self.time = time.time()

        self.device = args.device

    def encode(self, src_seq, src_mask):
        self.time = time.time()

        enc_out, enc_mask, _ = self.model.encoder(src_seq, src_mask)
        enc_out2 = self.model.project2(enc_out)
        self.lock.acquire()
        enc_mem = self.enc_mem

        return enc_out, enc_out2, enc_mask, *enc_mem 

    def decode(self, enc, seq):
        enc = list(enc)
        enc[0] = enc[0].expand(seq.size(0), -1, -1)
        if len(enc)==7:
            enc[1] = enc[1].expand(seq.size(0), -1, -1)
        enc[2] = enc[2].expand(seq.size(0), -1)

        return self.model.decode(seq, enc)

    def get_attn(self, enc, tgt_seq):
        return self.model.decoder(tgt_seq, enc[0], enc[2])[1]

    def get_logit(self, enc, tgt_seq):
        logit = self.model.decoder(tgt_seq, enc[0], enc[2])[0]
        return torch.log_softmax(logit, -1)

    def get_best_memory_entry(self, enc, tgt_seq):
        enc_out, enc_out2, enc_mask, tgt_emb_mem, tgt_mask_mem, enc_out_mem, enc_out_mem_mean = enc
        _, mem_attn_outs = self.model.decoder_mem(tgt_seq, enc_out2, enc_mask, tgt_emb_mem, tgt_mask_mem, enc_out_mem, enc_out_mem_mean)
        return mem_attn_outs[-1].argmax(-1)

    def new_words(self, words):
        if words != self.words:
            with self.lock:
                self.words = words

                if len(self.words)>0:
                    tgt_ids_mem = self.words_to_tensor()
                else:
                    tgt_ids_mem = torch.ones(1,2,dtype=torch.int64,device=self.tgt_ids_mem.device)
                    tgt_ids_mem[:,1] = 2
            
                self.enc_mem = self.model.encode_memory(tgt_ids_mem)
                print("!!!!! Updated memory encoding !!!!!")

    def words_to_tensor(self):
        bpes = [[1] + [x + 2 for x in self.sp.encode_as_ids(w.lower())] + [2] for w in self.words]

        tmp = torch.zeros(len(bpes), max([len(x) for x in bpes]), dtype=torch.int64)
        for i, word in enumerate(bpes):
            tmp[i, :len(word)] = torch.as_tensor(word)
        return tmp.to(self.device)

def init_punct_model(args):
    device = torch.device(args.device)

    mdic = torch.load(args.punct_dic)
    model = load_object(mdic['class'], mdic['module'], mdic['params'])
    model = model.to(device)
    model.load_state_dict(mdic['state'])
    model.eval()
    if args.fp16: model.half()

    return model

def clean_noise(seq, best_memory_entry, dic, space):
    clean_seq = []
    clean_bme = []
    word, has_noise, bmes = [], False, []
    for el,bme in zip(seq,best_memory_entry):
        token = dic[el-2]
        if token.startswith(space):
            if not has_noise:
                clean_seq.extend(word)
                clean_bme.extend(bmes)
            word = [el]
            bmes = [bme]
            has_noise = (el == 3 or el == 4) # noise or unknown
        else:
            word.append(el)
            bmes.append(bme)
            if el == 3 or el == 4: has_noise = True
    if not has_noise:
        clean_seq.extend(word)
        clean_bme.extend(bmes)
    return clean_seq, clean_bme

def token2punct(model, device, seg, lctx, rctx, dic, space):
    seq = seg.hypo
    bmes = seg.bmes

    if len(seq) == 0: return []

    puncts = {1:'', 2:'.', 3:',', 4:'?', 5:'!', 6:':', 7:';'}
    src = torch.LongTensor(lctx + seq + rctx)
    src = (src - 2).unsqueeze(0).to(device)
    mask = src.gt(0)
    out = torch.softmax(model(src, mask)[0], -1)  
    pred = torch.argmax(out.squeeze(0), -1).tolist()
    pred = pred[len(lctx):]
    if len(rctx) > 0: pred = pred[:-(len(rctx))]

    word_rest = ""
    hypo, tokens, bmes2 = [], [], []
    for j, (el,bme) in enumerate(zip(seq,bmes)):
        token = dic[el-2]
        if token.startswith(space) and len(tokens) > 0:
            word, norm = ''.join(tokens), pred[j-1]

            memoryUsed = False
            if word==word_rest.lower()[:len(word)]:
                word = word_rest[:len(word)]
                word_rest = word_rest[len(word)+1:]
                memoryUsed = True
            else:
                ids = set(bmes2)
                for id in ids:
                    if 0<id<=len(model.model.words):
                        word2 = model.model.words[id-1]
                        if word==word2.lower()[:len(word)]:
                            word = word2[:len(word)]
                            word_rest = word2[len(word)+1:]
                            memoryUsed = True
                            break
                if not memoryUsed:
                    word_rest = ""

            if norm > 7:
                if not memoryUsed:
                    word = word.capitalize()
                norm -= 7
            if norm > 1 and word_rest == "":
                word += puncts[norm]
            hypo.append(word)
            tokens = []
            bmes2 = []
        tokens.append(token[1:] if token.startswith(space) else token)
        bmes2.append(bme)

    if len(tokens) > 0:
        word, norm = ''.join(tokens), pred[j]
        
        memoryUsed = False
        if word==word_rest.lower()[:len(word)]:
             word = word_rest[:len(word)]
             word_rest = word_rest[len(word)+1:]
             memoryUsed = True
        else:
            ids = set(bmes2)
            for id in ids:
                if 0<id<=len(model.model.words):
                    word2 = model.model.words[id-1]
                    if word==word2.lower()[:len(word)]:
                        word = word2[:len(word)]
                        word_rest = word2[len(word)+1:]
                        memoryUsed = True
                        break
            if not memoryUsed:
                word_rest = ""

        if norm > 7:
            if not memoryUsed:
                word = word.capitalize()
            norm -= 7
        if norm > 1 and word_rest == "":
            word += puncts[norm]
        hypo.append(word)

    return hypo
