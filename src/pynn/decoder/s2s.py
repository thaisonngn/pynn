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

    def seq(self, rank):
        rank = min(rank, len(self.seqs)-1)
        return self.seqs[rank]

    def score(self, rank):
        rank = min(rank, len(self.scores)-1)
        return self.scores[rank]

    def best(self):
        return self.seqs[0], self.probs[0]

    def hypo(self, rank):
        hypo, prob = [], []
        for tk, p in zip(self.seqs[rank], self.probs[rank]):
            hypo.append(tk)
            prob.append(p)
            if tk == 2: break
        return hypo, prob

    def best_hypo(self):
        return self.hypo(0)

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
        if node >= len(self.scores): return
        
        base_score = self.scores[node]
        if self.seqs[node][-1] == 2 or tokens is None:
            acc_score = base_score + self.bases[node]
            self.adv_scores.append((base_score, acc_score, node, 0., 2))
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
        dec_out, attn, hid_cell = model.decode(enc, mask, seq, hid_cell)
        
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
        init_state=[1], len_norm=False, coverage=0., lm=None, lm_scale=0.5):
    batch_size = src_seq.size(0)
    enc_out, enc_mask = model.encode(src_seq, src_mask)[0:2]

    beams = [Beam(max_node, init_state, len_norm) for i in range(batch_size)]
    for step in range(max_len):
        l = 1 if step == 0 else max_node
        for k in range(l):
            seq = [beam.seq(k) for beam in beams]
            seq = torch.LongTensor(seq).to(device)

            dec_out, attn = model.decode(enc_out, enc_mask, seq)[0:2]
            if coverage > 0.:
                scores = model.coverage(enc_out, enc_mask, seq, attn)
                scores = scores.cpu().numpy()
                for beam, s in zip(beams, scores): beam.update(k, coverage*s)
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
        init_state=[1], len_norm=False, coverage=0., lm=None, lm_scale=0.5):
    batch_size = src_seq.size(0)
    enc_out, enc_mask = model.encode(src_seq, src_mask)[0:2]

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
            
            dec_out, attn, hid_cell = model.decode(enc_out, enc_mask, seq, hid_cell)
            if coverage > 0.:
                scores = model.coverage(enc_out, enc_mask, seq, attn)
                scores = scores.cpu().numpy()
                for beam, s in zip(beams, scores): beam.update(k, coverage*s)            
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

def beam_search_dual(model, src_seq, src_mask, device, dic, wdic, max_node=10, max_len=200,
        init_state=[1], len_norm=False, coverage=0.2, lm=None, lm_scale=0.5):
    batch_size = src_seq.size(0)
    enc_out, enc_mask = model.encode(src_seq, src_mask)[0:2]

    beams = [Beam(max_node, init_state, len_norm) for i in range(batch_size)]
    for step in range(max_len):
        l = 1 if step == 0 else max_node
        for k in range(l):
            seq = [beam.seq(k) for beam in beams]
            seq = torch.LongTensor(seq).to(device)

            dec_out, attn = model.decode(enc_out, enc_mask, seq)[0:2]
            if coverage > 0.:
                scores = model.coverage(enc_out, enc_mask, seq, attn)
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

    hypos = [[] for i in range(batch_size)]
    for j in range(max_node):
        seqs = [beam.hypo(j)[0] for beam in beams]
        
        wseqs, wb = [], 3
        for seq in seqs:
            words, tokens = [], []
            for tk in seq[1:]:
                if tk == 2: break
                if tk == wb:
                    words.append(' '.join(map(str, tokens)))
                    tokens = []
                else:
                    tokens.append(tk-2)
            if len(tokens) > 0: words.append(' '.join(map(str, tokens)))
            wseqs.append([[w+2 for w in dic.get(word, [1])] + [2] for word in words ])
        
        l = max(len(seq) for seq in seqs)
        seqs = [seq + [0]*(l-len(seq)) for seq in seqs]
        seqs = torch.LongTensor(seqs).to(device)
        emb_out = model.decode(enc_out, enc_mask, seqs)[2]
        emb_mask = seqs.gt(0)
        
        beams_ex = [Beam(max_node, [1], len_norm) for i in range(batch_size)]
        for step in range(max_len):
            l = 1 if step == 0 else max_node
            for k in range(l):
                seq = [beam.seq(k) for beam in beams_ex]
                seq = torch.LongTensor(seq).to(device)

                dec_out = model.trancode(enc_out, enc_mask, emb_out, emb_mask, seq)
                dec_out = dec_out[:,-1,:].squeeze(1).cpu().numpy()
                for beam, wseq, dout in zip(beams_ex, wseqs, dec_out):
                    if step < len(wseq):
                        tokens = wseq[step]
                        probs = [dout[tk] for tk in tokens]
                    else:
                        tokens, probs = [2], [0.]
                    for tk in tokens: beam.advance(k, probs, tokens)

            br = True
            for beam in beams_ex:
                beam.prune()
                br = False if not beam.done else br
            if br: break

        for hypo, beam in zip(hypos, beams_ex):
            hypo.append((beam.hypo(0), beam.score(0)))

    tokens = []
    for hypo in hypos:
        hypo.sort(key=lambda e : -e[1])
        hy, pb = hypo[0][0]
        tokens.append(hy[1:])
        
    return tokens

def search_and_rescore(model, src_seq, src_mask, device, dic, wdic, max_node=10, max_len=200, 
        wb=3, init_state=[1], len_norm=False, lm=None, lm_scale=0.5):
    batch_size = src_seq.size(0)
    enc_out, enc_mask = model.encode(src_seq, src_mask)[0:2]

    beams = [Beam(max_node, init_state, len_norm) for i in range(batch_size)]
    for step in range(max_len):
        l = 1 if step == 0 else max_node
        for k in range(l):
            seq = [beam.seq(k) for beam in beams]
            seq = torch.LongTensor(seq).to(device)

            dec_out = model.decode(enc_out, enc_mask, seq)[0]
            probs, tokens = dec_out.topk(max_node, dim=1)
            probs, tokens = probs.cpu().numpy(), tokens.cpu().numpy()

            for beam, prob, token in zip(beams, probs, tokens):
                beam.advance(k, prob, token)

        br = True
        for beam in beams:
            beam.prune()
            br = False if not beam.done else br
        if br: break
   
    hypos = []
    for j, beam in enumerate(beams):
        emb_seq, tgt_seq = [], []
        print('collecing hypo')
        for bseq in beam.seqs:
            eseq, words, tokens = [], [], []
            for tk in bseq[1:]:
                if tk == 2: break
                eseq.append(tk)
                if tk == wb:
                    words.append(tokens)
                    tokens = []
                else:
                    tokens.append(tk-2)
            if len(tokens) > 0: words.append(tokens)
            tseq = [[]]
            for tokens in words:
                new_seq, word = [], ' '.join(map(str, tokens))
                for w in dic.get(word, [1]):
                    for s in tseq: new_seq.append(s + [w+2])
                tseq = new_seq
            tgt_seq.extend(tseq)
            emb_seq.extend([eseq] * len(tseq))

        e_out, e_mask = enc_out[j].unsqueeze(0), enc_mask[j].unsqueeze(0)
        print('rescoring %d' % len(tgt_seq))
        tgt_out, start, size = [], 0, 320
        while start < len(tgt_seq):
            seqs = emb_seq[start: start+size]
            bs = len(seqs)
            enc, mask = e_out.expand(bs, -1, -1), e_mask.expand(bs, -1)

            l = max(len(seq) for seq in seqs)
            seqs = [[1] + seq + [2] + [0]*(l-len(seq)) for seq in seqs]
            seqs = torch.LongTensor(seqs).to(device)
            emb_out = model.decode(enc, mask, seqs)[1]
            emb_mask = seqs.gt(0)
        
            seqs = tgt_seq[start: start+bs]
            l = max(len(seq) for seq in seqs)
            seqs = [[1] + seq + [2] + [0]*(l-len(seq)) for seq in seqs]
            seqs = torch.LongTensor(seqs).to(device)
            #if start == 0: print(seqs) 
            out = model.trancode(enc, mask, emb_out, emb_mask, seqs)
            tgt_out.extend(out.tolist())
            start += size
        print('selecting best hypo')
        lst = []
        for i, seq in enumerate(tgt_seq):
            #print(' '.join(wdic[w-2][1:] for w in seq))
            #print([tgt_out[i][t][k] for t,k in enumerate(seq)])
            score = sum([tgt_out[i][t][k] for t,k in enumerate(seq + [2])])
            lst.append((score, i))
        lst.sort(key=lambda e : -e[0])
        hypos.append(tgt_seq[lst[0][1]] + [2])
    #print(hypos)
    return hypos
