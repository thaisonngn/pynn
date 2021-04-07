#!/usr/bin/python

from datetime import date
import numpy
import torch

from pynn.util import audio
from pynn.decoder.s2s import Beam
from pynn.util import load_object

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
        hypo = ['<unk>' if w.startswith('%') or w.startswith('*') or w.startswith('<') or \
                w.startswith('-') or w.endswith('-') else w for w in hypo]
        words, pw = [], ''
        for w in hypo:
            if w == '<unk>' and pw == w: continue
            words.append(w)
            pw = w
        hypo = words

    return hypo

def incl_search(model, src, max_node=8, max_len=10, states=[1], len_norm=False, prune=1.0):
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
    
    if args.fp16: model.half()

    if args.int8:
        model = torch.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.LSTM, torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8)  # the target dtype for quantized weights
    #print(model)
    return model, device, dic

def decode(model, device, args, adc, fbank_mat, start=0, prefix=[1]):
    signal = numpy.frombuffer(adc[start*16*10*2:], numpy.int16)
    feats = audio.extract_fbank(signal, fbank_mat, sample_rate=16000)
    feats = feats - feats.mean(axis=0, keepdims=True)
     
    frames = (feats.shape[0])
    print("Decoding for audio segment of %d frames" % frames)
    expired = date(2021, 6, 20)
    if frames < 10 or date.today() > expired: return [], None, None, frames
    
    space, beam_size, max_len = args.space, args.beam_size, args.max_len
    win, stable_time = args.incl_block, args.stable_time
    head, padding = args.attn_head, args.attn_padding

    with torch.no_grad():
        src = torch.HalfTensor(feats) if args.fp16 else torch.FloatTensor(feats)
        src = src.to(device)
        enc_out, mask, hypo, score, sth = incl_search(model, src, beam_size, max_len, prefix)

        tgt = torch.LongTensor(hypo).to(device).view(1, -1)
        attn = model.get_attn(enc_out, mask, tgt)
        attn = attn[0]
        cs = torch.cumsum(attn[head], dim=1)
        ep = cs.le(1.-padding).sum(dim=1)
        ep = ep.cpu().numpy() * 4
        sp = sp = cs.le(padding).sum(dim=1)
        sp = sp.cpu().numpy() * 4

    return hypo, sp, ep, frames

def init_punct_model(args):
    device = torch.device(args.device)

    mdic = torch.load(args.punct_dic)
    model = load_object(mdic['class'], mdic['module'], mdic['params'])
    model = model.to(device)
    model.load_state_dict(mdic['state'])
    model.eval()

    if args.fp16: model.half()

    if args.int8:
        model = torch.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.LSTM, torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8)  # the target dtype for quantized weights
    #print(model)
    return model

def clean_noise(seq, dic, space):
    clean_seq = []
    word, has_noise = [], False
    for el in seq:
        token = dic[el-2]
        if token.startswith(space):
            if not has_noise: clean_seq.extend(word)
            word = [el]
            has_noise = (el == 3 or el == 4) # noise or unknown
        else:
            word.append(el)
            if el == 3 or el == 4: has_noise = True
    if not has_noise: clean_seq.extend(word)
    return clean_seq

def token2punct(model, device, seq, lctx, rctx, dic, space):
    if len(seq) == 0: return []

    puncts = {1:'', 2:'.', 3:',', 4:'?', 5:'!', 6:':', 7:';'}
    src = torch.LongTensor(lctx + seq + rctx)
    src = (src - 2).unsqueeze(0).to(device)
    mask = src.gt(0)
    out = torch.softmax(model(src, mask)[0], -1)  
    pred = torch.argmax(out.squeeze(0), -1).tolist()
    pred = pred[len(lctx):]
    if len(rctx) > 0: pred = pred[:-(len(rctx))]

    hypo, tokens = [], []
    for j, el in enumerate(seq):
        token = dic[el-2]
        if token.startswith(space) and len(tokens) > 0:
            word, norm = ''.join(tokens), pred[j-1]
            if norm > 7:
                word = word.capitalize()
                norm -= 7
            if norm > 1:
                word += puncts[norm]
            hypo.append(word)
            tokens = []
        tokens.append(token[1:] if token.startswith(space) else token)

    if len(tokens) > 0:
        word, norm = ''.join(tokens), pred[j]
        if norm > 7:
            word = word.capitalize()
            norm -= 7
        if norm > 1:
            word += puncts[norm]
        hypo.append(word)
    return hypo
