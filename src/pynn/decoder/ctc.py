# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import numpy as np

import torch

class Beam(object):
    def __init__(self, beam_size=10, pruning=1.5, blank=0, blank_scale=1.):
        self.beam_size = beam_size
        self.pruning_size = int(beam_size*pruning)
        self.blank = blank
        self.blank_scale = blank_scale
        self.seqs = [[[], 0.]]
        self.index = {'': 0}
        self.paths = [(0, 0., 0., blank)]
        self.adv_paths = []

    def advance(self, probs, tokens):
        blank, blank_scale, adv_paths = self.blank, self.blank_scale, []
        for prob, tk in zip(probs, tokens):
            prob = prob if tk!=blank else prob*blank_scale
            nodes = [[sid, pscore+prob, ascore+prob, tk, pre_tk]
                     for sid, pscore, ascore, pre_tk in self.paths]
            adv_paths.extend(nodes)
        adv_paths.sort(key=lambda e : -e[2])
        self.adv_paths = adv_paths[:self.pruning_size]

        new_seqs, new_idx = [], []
        for path in self.adv_paths:
            sid, pscore, ascore, tk, pre_tk = path
            if tk == pre_tk or tk == blank: continue
            seq = self.seqs[sid][0] + [tk]
            key = '_'.join(map(str, seq))
            if key in self.index:
                sid = self.index[key]
                path[2] += self.seqs[sid][1]
            else:
                self.seqs.append([seq, 0.])
                sid = len(self.seqs) - 1
                self.index[key] = sid
                new_seqs.append(seq)
                new_idx.append(sid)
            path[0] = sid

        self.new_seqs = new_seqs
        self.new_idx = new_idx

    def find_new_seqs(self):
        return self.new_seqs, self.new_idx

    def score_new_seqs(self, seq_idx, scores):
        for sid, score in zip(seq_idx, scores):
            self.seqs[sid][1] += score
            for path in self.adv_paths:
                if path[0] == sid: path[2] += score

    def prune(self):
        self.adv_paths.sort(key=lambda e : -e[2])
        paths, adv_paths = [], self.adv_paths[:self.beam_size]
        for sid, pscore, ascore, tk, *_ in adv_paths:
            paths.append((sid, pscore, ascore, tk))
        self.paths = paths 

    def best(self, topk=1):
        return [self.seqs[sid][0] for sid, *_ in self.paths[:topk]]

def beam_search(model, src, mask, device, lm=None, lm_scale=0., beam_size=10,
        pruning=1.5, blank=0, blank_scale=1.):
    logits, mask = model.decode(src, mask)
    lens = mask.sum(-1).cpu().numpy()
    b_probs, b_tokens = logits.transpose(0, 1).topk(beam_size, dim=-1)
    b_probs, b_tokens = b_probs.cpu().numpy(), b_tokens.cpu().numpy()

    b_beams = [Beam(beam_size, pruning, blank, blank_scale) for i in range(logits.size(0))]
    for t, (probs, tokens) in enumerate(zip(b_probs, b_tokens)):
        beams = []
        for l, beam in zip(lens, b_beams):
            if t < l: beams.append(beam)

        new_seqs, new_idx = [], []
        for j, beam in enumerate(beams):
            beam.advance(probs[j], tokens[j])
            seqs, idx = beam.find_new_seqs()
            new_seqs.extend(seqs)
            new_idx.append(idx)

        if len(new_seqs) > 0 and lm is not None and lm_scale > 0.:
            l = max(len(seq) for seq in new_seqs)
            lm_seqs = [[-1+blank] + seq[:-1] + [-2+blank]*(l-len(seq)) for seq in new_seqs]
            lm_seqs = torch.LongTensor(lm_seqs) + 2 - blank
            lm_prob = lm(lm_seqs.to(device))
            lm_prob = torch.log_softmax(lm_prob, -1).cpu().numpy()

            lm_scores = [lm_prob[j, len(seq)-1, seq[-1]+2-blank] * lm_scale \
                         for j, seq in enumerate(new_seqs)]
            start_i = 0
            for j, (beam, idx) in enumerate(zip(beams, new_idx)):
                if len(idx) == 0: continue
                beam.score_new_seqs(idx, lm_scores[start_i: start_i+len(idx)])
                start_i += len(idx)
        for beam in beams: beam.prune()
    hypos = [beam.best(topk=1)[0] for beam in b_beams]
    return hypos

def greedy_search(logits, blank=0):
    seqs = torch.argmax(logits, -1).cpu().numpy()
    hypos = []
    for seq in seqs:
        hypo, prev = [], -1
        for pred in seq:
            if pred != prev and pred != blank:
                hypo.append(pred)
                prev = pred
            if pred == blank: prev = -1
        hypos.append(hypo)
    return hypos

def greedy_align(probs, tokens):
    mp = torch.argmax(probs, -1).tolist()
    probs, tokens = probs.tolist(), tokens.tolist()

    hypo = []
    j, tk, start = 0, tokens[0], -1
    for i, frame in enumerate(probs):
        #print('%d %d %d %f' % (j, i, mp[i], frame[tk]))
        if tk == mp[i]:
            if start == -1: start = i
            continue
        if start > -1:
            hypo.append((tk, start, i))
            j += 1
            if j == len(tokens): break
            tk = tokens[j]
            start = i if tk == mp[i] else -1

    if start > -1 and j < len(tokens): hypo.append((tk, start, i))
    return hypo

def viterbi_align(probs, tokens, beam_size=10, blank=0, blank_scale=1., strict=False):
    probs, tokens = probs.numpy(), tokens.tolist()
    paths = [(0.0, blank, 0, [])]
    for i, prob in enumerate(probs):
        adv_paths = []
        for score, tk, j, seq in paths:
            bj = j if tk==blank else j+1
            adv_paths.append((score+prob[blank]*blank_scale, blank, bj, seq+[-1]))
            if tk != blank:
                adv_paths.append((score+prob[tk], tk, j, seq+[j]))
            if bj < len(tokens):
                token = tokens[bj]
                adv_paths.append((score+prob[token], token, bj, seq+[bj]))

        if len(adv_paths) > beam_size:
            adv_paths.sort(key=lambda e : -e[0])
        paths = adv_paths[:beam_size]

    scores, l = [], len(tokens)
    for score, tk, j, seq in paths:
        if strict:
            if j == l: scores.append((score, seq))
        else:
            scores.append((score, seq))
    alg = []
    if len(scores) > 0:
        scores.sort(key=lambda e : -e[0])
        pj, start = -1, -1
        for i, j in enumerate(scores[0][1]):
            if j == -1:
                if pj > -1: alg.append((tokens[pj], start, i))
                pj = -1
            elif pj != j:
                if pj > -1: alg.append((tokens[pj], start, i))
                pj, start = j, i
        if pj > -1: alg.append((tokens[pj], start, i))
    return alg
