# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import numpy as np
import torch

class Beam(object):
    def __init__(self, max_node=10, init_state=[1], len_norm=True):
        self.max_node = max_node
        self.len_norm = len_norm       
        self.adv_scores = []
        self.bases = [0.0]
        self.scores = [0.0]
        self.seqs = [init_state]
        self.probs = [[1.0]]
        self.done = False

    def seq(self, node):
        return self.seqs[node]
    
    def best(self):
        return self.seqs[0], self.probs[0]

    def best_hypo(self):
        hypo, prob = [], []
        for tk, p in zip(self.seqs[0], self.probs[0]):
            hypo.append(tk)
            prob.append(p)
            if tk == 2: break
        return hypo, prob

    def update(self, node, score):
        self.bases[node] = score

    def advance(self, node, probs, tokens):
        base_score = self.scores[node]
        if self.seqs[node][-1] == 2 or tokens is None:
            acc_score = base_score + self.bases[node]
            self.adv_scores.append((base_score, acc_score, node, 1.0, 2))
            return
        l = len(self.seqs[0])
        for prob, token in zip(probs, tokens):
            if self.len_norm:
                score = (base_score*l + prob) / (l+1)
            else:
                score = base_score + prob
            acc_score = score + self.bases[node]
            self.adv_scores.append((score, acc_score, node, prob, token))

    def prune(self):
        self.adv_scores.sort(key=lambda e : -e[1])
        
        new_scores = []
        new_seqs = []
        new_probs = []
        done = True
        for j in range(len(self.adv_scores)):
            score, acc_score, node, prob, token = self.adv_scores[j]
            new_scores.append(score)
            new_seqs.append(self.seqs[node] + [token])
            new_probs.append(self.probs[node] + [prob])
            if token != 2: done = False

        self.done = done
        self.scores = new_scores
        self.bases = [0.] * len(new_scores)
        self.seqs = new_seqs
        self.probs = new_probs
        self.adv_scores = []

class InclHypo(object):
    def __init__(self, stable_time=25, approx=1):
        self.stable_time = stable_time
        self.approx = approx

        self.fhypo = []
        self.shypo = []
        
    def stable_hypo(self):
        return self.shypo

    def full_hypo(self):
        return self.fhypo
        
    def update(self, hypo, now):
        shypo = []
        st = self.stable_time
        ap = self.approx
        for i in range(min(len(self.fhypo), len(hypo))):
            ft, fs, fe = self.fhypo[i]
            ht, hs, he = hypo[i]
            if ft == ht and fs-ap <= hs <= fs+ap and he-ap <= fe <= fe+ap and he+st<now:
                shypo.append(ht)
            else:
                break
        self.shypo = shypo
        self.fhypo = hypo
        

def beam_search(model, src_seq, src_mask, device, max_node=10, max_len=200, 
        init_state=[1], len_norm=True, coverage=0., lm=None, lm_scale=0.5):
    batch_size = src_seq.size(0)    
    enc_out, src_mask = model.encode(src_seq, src_mask)

    beams = [Beam(max_node, init_state, len_norm) for i in range(batch_size)]
    for step in range(max_len):
        l = 1 if step == 0 else max_node
        for k in range(l):
            seq = [beam.seq(k) for beam in beams]
            seq = torch.LongTensor(seq).to(device)

            dec_out = model.decode(enc_out, src_mask, seq)
            if lm is not None:
                lm_out = lm.decode(seq)
                dec_out += lm_out * lm_scale

            if coverage > 0.:
                scores = model.converage(enc_out, src_mask, seq)
                scores = scores.cpu().numpy()
                for beam, s in zip(beams, scores): beam.update(k, coverage*s)

            probs, tokens = dec_out.topk(max_node, dim=1)
            probs, tokens = probs.cpu().numpy(), tokens.cpu().numpy()

            for beam, prob, token in zip(beams, probs, tokens):
                beam.advance(k, prob, token)

        br = True
        for beam in beams:
            beam.prune()
            br = False if not beam.done else br
        if br: break

    tokens = np.zeros((batch_size, step), dtype="int32")
    probs = np.zeros((batch_size, step), dtype="float32")
    for i, beam in enumerate(beams):
        hypo, prob = beam.best()
        tokens[i,:] = hypo[1:step+1]
        probs[i,:] = prob[1:step+1]

    return tokens, probs

