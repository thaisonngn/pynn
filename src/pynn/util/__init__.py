# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math
import numpy as np

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
