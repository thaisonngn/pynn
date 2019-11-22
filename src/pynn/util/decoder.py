import math
import numpy as np
import torch

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
 
    pre_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        cur_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = pre_row[j + 1] + 1
            dels = cur_row[j] + 1
            subs = pre_row[j] + (c1 != c2)
            cur_row.append(min(ins, dels, subs))
        pre_row = cur_row
 
    return pre_row[-1]

def compute_ter(hypos, refs):
    err = 0
    l = 0
    for hypo, ref in zip(hypos, refs):
        err += levenshtein(hypo, ref)
        l += len(ref)
    ter = float(err) / l
    return ter

def parse_time_info(utt):
    stime, etime =  utt.split('-')[-2:]
    conv = utt[:-len(stime)-len(etime)-2]
    stime = stime[:-2].lstrip('0') + '.' + stime[-2:]
    etime = etime[:-2].lstrip('0') + '.' + etime[-2:]
    return (conv, stime, etime)

class Beam(object):
    def __init__(self, max_node=10, init_state=[1], len_norm=True):
        self.max_node = max_node
        self.len_norm = len_norm       
        self.adv_scores = []
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
        self.scores[node] += score

    def advance(self, node, probs, tokens):
        base_score = self.scores[node]
        if self.seqs[node][-1] == 2 or tokens is None:
            self.adv_scores.append((base_score, node, 1.0, 2))
            return
        l = len(self.seqs[0])
        for prob, token in zip(probs, tokens):
            if self.len_norm:
                total_score = (base_score*l + prob) / (l+1)
            else:
                total_score = base_score + prob
            self.adv_scores.append((total_score, node, prob, token))

    def prune(self):
        self.adv_scores.sort(key=lambda e : -e[0])
        
        new_scores = []
        new_seqs = []
        new_probs = []
        done = True
        for j in range(len(self.adv_scores)):
            total_score, node, prob, token = self.adv_scores[j]
            new_scores.append(total_score)
            new_seqs.append(self.seqs[node] + [token])
            new_probs.append(self.probs[node] + [prob])
            if token != 2: done = False

        self.done = done
        self.scores = new_scores
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
        
class Decoder(object):
    @staticmethod
    def ter(hypo, ref):
        return levenshtein(hypo, ref) * 1.0 / len(ref)

    @staticmethod
    def score(hypo, ref):
        err = levenshtein(hypo, ref)
        l = len(ref)
        return (err, l, err*1.0/l) 

    @staticmethod
    def decode(probs, blank_index=0):
        max_probs = torch.argmax(probs, -1)
        sequences = max_probs.view(max_probs.size(0), max_probs.size(1))
        hypos = []
        for j in range(len(sequences)):
            hypo = []
            seq = sequences[j]
            prev = -1
            for i in range(len(seq)):
                pred = int(seq[i])
                if pred != prev and pred != blank_index:
                    hypo.append(pred)
                    prev = pred
            hypos.append(hypo)

        return hypos
        
    @staticmethod
    def beam_search(model, src_seq, src_mask, device, max_node=10, max_len=200, 
            init_state=[1], len_norm=True, alg_scale=0., lm=None, lm_scale=0.5):
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
                if alg_scale > 0.:
                    lens = model.align(enc_out, src_mask, seq)
                    lens = lens.cpu().numpy()
                    for beam, ln in zip(beams, lens): beam.update(alg_scale * ln)

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


def write_ctm(hypos, scores, ctm, utts, dic, word_dic=None, space=''):
    for i in range(len(hypos)):
        ctm.write('# %s\n' % utts[i])
        conv, stime, etime = parse_time_info(utts[i])
        hypo = []
        pw, ps = '', 0.
        for wid, score in zip(hypos[i], scores[i]):
            if wid == 2:
                if pw != '': hypo.append((pw, ps))
                break
            token = dic[wid-2]
            if space == '':
                hypo.append((token, score))
            else:
                if token.startswith(space):
                    if pw != '':  hypo.append((pw, ps))
                    if token != space:
                        pw, ps = token.replace(space, ''), score
                    else:
                        pw, ps = '', 0.
                else:
                    pw += token
                    ps += score

        if word_dic is not None:
            hypo = [(word_dic.get(word,'<unk>'), score) for word, score in hypo]

        if len(hypo) == 0: continue
        stime = float(stime)
        span = (float(etime) - stime) / len(hypo)
        for word, score in hypo:
            if word == '' or word == ' ': continue
            score = math.exp(score)
            ctm.write('%s 1 %.2f %.2f %s \t %.2f\n' % (conv, stime, span-0.01, word, score))
            stime += span

def write_text(hypos, scores, fout, utts, dic=None, space=''):
    for i in range(len(hypos)):
        hypo = []
        pw = ''
        for tid in hypos[i]:
            if tid == 2:
                if pw != '': hypo.append(pw)
                break
            token = str(tid-2) if dic is None else dic[tid-2]
            if space == '':
                hypo.append(token)
            else:
                if token.startswith(space):
                    if pw != '':  hypo.append(pw)
                    pw = token.replace(space, '') if token != space else ''
                else:
                    pw += token
        hypo = ' '.join(hypo)
        fout.write('%s %s\n' % (utts[i], hypo))

