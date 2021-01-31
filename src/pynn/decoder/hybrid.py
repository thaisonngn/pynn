# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import numpy as np

import torch
from .ctc import Beam

def beam_search(model, src, mask, device, s2s_scale=1.,
        lm=None, lm_scale=0., beam_size=10, pruning=1.5, blank=0):
    enc_out, enc_mask = model.encode(src, mask)[0:2]
    lens = enc_mask.sum(-1).cpu().numpy()
    
    logits = model.decode_ctc(enc_out, enc_mask)
    logits.index_fill_(-1, torch.tensor([1, 2]).to(device), -np.inf)
    b_probs, b_tokens = logits.transpose(0, 1).topk(beam_size, dim=-1)
    b_probs, b_tokens = b_probs.cpu().numpy(), b_tokens.cpu().numpy()
    batch_size = logits.size(0)

    b_beams = [Beam(beam_size, pruning, blank) for i in range(batch_size)]
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

        if len(new_seqs) > 0:
            sl = max(len(seq) for seq in new_seqs)
            sb = max(len(idx) for idx in new_idx)

            if s2s_scale > 0.:
                seqs = np.zeros((batch_size*sb, sl+1), dtype=int)
                start_i = 0
                for j, idx in enumerate(new_idx):
                    for i in range(len(idx)):
                        seq = new_seqs[start_i + i]
                        seqs[j*sb + i, :len(seq)+1] = [1] + seq[:-1]  + [2]
                    start_i += len(idx)
                seqs = torch.LongTensor(seqs).to(device)
                out = enc_out.unsqueeze(1).expand(-1, sb, -1, -1).contiguous()
                out = out.view(-1, out.size(2), out.size(3))
                mask = enc_mask.unsqueeze(1).expand(-1, sb, -1).contiguous()
                mask = mask.view(-1, mask.size(2))
                s2s_prob = model.decode_s2s(out, mask, seqs).cpu().numpy()
                start_i = 0
                for j, (beam, idx) in enumerate(zip(beams, new_idx)):
                    scores = []
                    for i in range(len(idx)):
                        seq = new_seqs[start_i + i]
                        score = s2s_prob[j*sb + i, len(seq)-1, seq[-1]]
                        scores.append(score * s2s_scale)
                    beam.score_new_seqs(idx, scores)
                    start_i += len(idx)

            if lm is not None and lm_scale > 0.:
                seqs = [[1] + seq[:-1] + [2]*(sl-len(seq)) for seq in new_seqs]
                seqs = torch.LongTensor(seqs).to(device)
                lm_prob = lm(seqs)
                lm_prob = torch.log_softmax(lm_prob, -1).cpu().numpy()
                lm_scores = [lm_prob[j, len(seq)-1, seq[-1]] * lm_scale \
                             for j, seq in enumerate(new_seqs)]
                start_i = 0
                for j, (beam, idx) in enumerate(zip(beams, new_idx)):
                    if len(idx) == 0: continue
                    beam.score_new_seqs(idx, lm_scores[start_i: start_i+len(idx)])
                    start_i += len(idx)

        for beam in beams: beam.prune()

    hypos = [beam.best(topk=1)[0] + [2] for beam in b_beams]
    return hypos

