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
        self.cache = [None]
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

    def stable_hypo(self, sr=1.):
        best = self.seqs[0]
        score, j = self.scores[0] * sr, 0
        for j in range(1, len(self.seqs)):
            if self.scores[j] > score: break

        for i in range(len(best)):
            tk = best[i]
            br = False
            for seq in self.seqs[1:j]:
                if seq[i] == tk: continue
                br = True; break
            if tk == 2 or br: break
        return best[:i]

    def update(self, node, score):
        self.bases[node] = score

    def advance(self, node, probs, tokens, node_cache=None):
        base_score = self.scores[node]
        if self.seqs[node][-1] == 2 or tokens is None:
            acc_score = base_score + self.bases[node]
            self.adv_scores.append((base_score, acc_score, node, 1.0, 2))
            return
        l = len(self.seqs[node])
        for prob, token in zip(probs, tokens):
            if self.len_norm:
                score = (base_score*l + prob) / (l+1)
            else:
                score = base_score + prob
            acc_score = score + self.bases[node]
            self.adv_scores.append((score, acc_score, node, prob, token))
        self.cache[node] = node_cache

    def prune(self):
        self.adv_scores.sort(key=lambda e : -e[1])
        
        new_scores = []
        new_seqs = []
        new_probs = []
        new_cache = []
        done = True
        for j in range(min(len(self.adv_scores), self.max_node)):
            score, acc_score, node, prob, token = self.adv_scores[j]
            new_scores.append(score)
            new_seqs.append(self.seqs[node] + [token])
            new_probs.append(self.probs[node] + [prob])
            new_cache.append(self.cache[node])
            if token != 2: done = False

        self.done = done
        self.scores = new_scores
        self.bases = [0.] * len(new_scores)
        self.seqs = new_seqs
        self.probs = new_probs
        self.cache = new_cache
        self.adv_scores = []

def partial_search(model, src, max_node=8, max_len=10, states=[1], len_norm=False, prune=1.0):
    enc_out = model.encode(src.unsqueeze(0), None)[0]
    enc_mask = torch.ones((1, enc_out.size(1)), dtype=torch.uint8).to(src.device)

    beam = Beam(max_node, [1], len_norm)
    if len(states) > 1:
        seq = torch.LongTensor(states).to(src.device).view(1, -1)
        logits = model.get_logit(enc_out, enc_mask, seq)
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

        if l > 1:
            cache = [beam.cache[k] for k in range(l)]
            hid, cell = zip(*cache)
            hid, cell = torch.stack(hid, dim=1), torch.stack(cell, dim=1)
            hid_cell = (hid, cell)
            seq = seq[:, -1].view(-1, 1)
        else:
            hid_cell = None

        enc = enc_out.expand(seq.size(0), -1, -1)
        mask = enc_mask.expand(seq.size(0), -1)
        dec_out, hid_cell = model.decode(enc, mask, seq, hid_cell)
        
        probs, tokens = dec_out.topk(max_node, dim=1)
        probs, tokens = probs.cpu().numpy(), tokens.cpu().numpy()

        hid, cell = hid_cell
        hid = [(hid[:,k,:].clone(), cell[:,k,:].clone()) for k in range(l)]

        for k in range(l):
            prob, token, cache = probs[k], tokens[k], hid[k]
            beam.advance(k, prob, token, cache)

        beam.prune()
        if beam.done: break
    hypo, prob = beam.best_hypo()
    sth = beam.stable_hypo(prune)

    return enc_out, enc_mask, hypo, prob, sth

def partial_search_multi(model, src, max_node=8, max_len=10, states=[1], len_norm=False, prune=1.0):
    mask = torch.ones((1, src.size(0)), dtype=torch.uint8).to(src.device)
    enc_out, enc_mask = model.encode(src.unsqueeze(0), mask)[0:2]

    beam = Beam(max_node, [1], len_norm)
    if len(states) > 1:
        seq = torch.LongTensor(states).to(src.device).view(1, -1)
        logits = model.get_logit(enc_out, enc_mask, seq)
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

        encs = [enc.expand(l, -1, -1) for enc in enc_out]
        masks = [mask.expand(l, -1) for mask in enc_mask]
        dec_out = model.decode(encs, masks, seq)[0]
        
        probs, tokens = dec_out.topk(max_node, dim=1)
        probs, tokens = probs.cpu().numpy(), tokens.cpu().numpy()

        for k in range(l):
            prob, token = probs[k], tokens[k]
            beam.advance(k, prob, token)

        beam.prune()
        if beam.done: break
    hypo, prob = beam.best_hypo()
    sth = beam.stable_hypo(prune)

    return enc_out, enc_mask, hypo, prob, sth

def beam_search(model, src_seq, src_mask, device, max_node=10, max_len=200, 
        init_state=[1], len_norm=True, lm=None, lm_scale=0.5):
    batch_size = src_seq.size(0)
    enc_out, src_mask = model.encode(src_seq, src_mask)[0:2]

    beams = [Beam(max_node, init_state, len_norm) for i in range(batch_size)]
    for step in range(max_len):
        l = 1 if step == 0 else max_node
        for k in range(l):
            seq = [beam.seq(k) for beam in beams]
            seq = torch.LongTensor(seq).to(device)

            dec_out = model.decode(enc_out, src_mask, seq)[0]
            if lm is not None:
                lm_out = lm.decode(seq)
                dec_out += lm_out * lm_scale

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

def beam_search_cache(model, src_seq, src_mask, device, max_node=10, max_len=200, 
        init_state=[1], len_norm=True, lm=None, lm_scale=0.5):
    batch_size = src_seq.size(0)
    enc_out, src_mask = model.encode(src_seq, src_mask)[0:2]

    beams = [Beam(max_node, init_state, len_norm) for i in range(batch_size)]
    for step in range(max_len):
        l = 1 if step == 0 else max_node
        for k in range(l):
            seq = [beam.seq(k) for beam in beams]
            seq = torch.LongTensor(seq).to(device)

            if l > 1:
                cache = [beam.cache[k] for beam in beams]
                hid, cell = zip(*cache)
                hid, cell = torch.stack(hid, dim=1), torch.stack(cell, dim=1)
                hid_cell = (hid, cell)
                seq = seq[:, -1].view(-1, 1)
            else:
                hid_cell = None
            
            dec_out, hid_cell = model.decode(enc_out, src_mask, seq, hid_cell)
            if lm is not None:
                lm_out = lm.decode(seq)
                dec_out += lm_out * lm_scale

            probs, tokens = dec_out.topk(max_node, dim=1)
            probs, tokens = probs.cpu().numpy(), tokens.cpu().numpy()

            hid, cell = hid_cell
            hid = [(hid[:,j,:].clone(), cell[:,j,:].clone()) for j in range(batch_size)]

            for beam, prob, token, h in zip(beams, probs, tokens, hid):
                beam.advance(k, prob, token, h)

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
