# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import math
import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from . import EpochPool, load_last_chkpt, save_last_chkpt
from . import ScheduledOptim, cal_ce_loss

def time_constraint_loss(attn, mask, sid, eid):
    T = attn.size(-1)    
    rg = torch.arange(T, device=mask.device).view(1, 1, 1, -1)
    rg = rg.expand(attn.size(0), attn.size(1), attn.size(2), -1)

    sid = sid.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, T)    
    sid = sid.gt(rg)

    eid = eid.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, T)
    eid = eid.le(rg)
    
    mask = mask.unsqueeze(1).unsqueeze(2)
    mask = mask.expand(-1, attn.size(1), attn.size(2), -1)
    #tid = (sid + eid + mask).gt(1)
    tid = (eid + mask).gt(1) # fwd
    #tid = (sid + mask).gt(1) # bwd

    attn = attn.masked_select(tid)
    #loss = torch.log(attn+0.1).sum() / tid.size(1)
    #loss = torch.log(attn+0.1).sum() / (tid.size(1)*tid.size(-1))
    #loss = attn.sum() / (tid.size(1)*tid.size(-1))
    loss = attn.sum()

    return loss

def train_epoch(model, data, opt, eps, device, batch_input, batch_update, n_print,
                loss_norm=False, grad_norm=True, grad_clip=0.):
    ''' Epoch operation in training phase'''
    model.train()
    
    total_loss = 0.; n_word_total = 0; n_word_correct = 0
    prints = n_print
    p_loss = 0.; p_total = 0; p_correct = 0;
    a_loss = 0.

    updates = 0
    n_seq = 0

    data.initialize()
    opt.zero_grad()
    while data.available():
        # prepare data
        batch = data.next(batch_input)
        seqs, last = batch[-2:]
        src_seq, src_mask, tgt_seq, sid, eid = map(lambda x: x.to(device), batch[:-2])

        gold = tgt_seq[:, 1:]
        tgt_seq = tgt_seq[:, :-1]
        n_seq += seqs

        try:
            # forward
            pred, attn, masks = model.attend(src_seq, src_mask, tgt_seq)
            # backward
            pred = pred.view(-1, pred.size(2))
            loss, loss_data, n_correct, n_total = cal_ce_loss(pred, gold, eps)
            attn_loss = time_constraint_loss(attn, masks, sid, eid)

            loss += attn_loss*0.2
            if torch.isnan(loss.data):
                print("    inf loss at %d" % n_seq); continue
            if loss_norm: loss = loss.div(n_total)
            opt.backward(loss)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print('    WARNING: ran out of memory on GPU at %d' % n_seq)
                torch.cuda.empty_cache(); continue
            raise err

        updates += n_total
        # update parameters
        if last or updates >= batch_update:
            if grad_clip > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            g_norm = updates if grad_norm else 1.
            opt.step_and_update_lr(updates)
            opt.zero_grad()
            updates = 0

        # note keeping
        total_loss += loss_data

        n_word_total += n_total;  n_word_correct += n_correct
        p_loss += loss_data; p_total += n_total; p_correct += n_correct;
        a_loss += attn_loss.data.item()
        
        if n_seq > prints:
            ppl = math.exp(min(p_loss/p_total, 100))
            pred = p_correct * 1. / p_total
            a_loss /= p_total
            print('    Seq: {:6d}, lr: {:.7f}, ppl: {:9.4f}, attn-loss: {:.3f}, '\
                    'updates: {:6d}, correct: {:.2f}'.format(n_seq, opt.lr, ppl, a_loss, opt.steps, pred))
            prints += n_print
            p_loss = 0.; p_total = 0; p_correct = 0
            a_loss = 0.
    
    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, data, device, batch_input):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    
    with torch.no_grad():
        data.initialize()
        while data.available():
            # prepare data
            batch = data.next(batch_input)
            src_seq, src_mask, tgt_seq = map(lambda x: x.to(device), batch[:-4])
            gold = tgt_seq[:, 1:]
            tgt_seq = tgt_seq[:, :-1]

            # forward
            pred = model(src_seq, src_mask, tgt_seq)[0]
            loss, loss_data, n_correct, n_total = cal_ce_loss(pred, gold)

            # note keeping
            total_loss += loss_data

            n_word_total += n_total
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy
    
def train_model(model, datasets, epochs, device, cfg,
                loss_norm=False, grad_norm=True, fp16=False):
    ''' Start training '''

    model_path = cfg['model_path']
    lr = cfg['lr']
    eps = cfg['smooth']

    n_warmup = cfg['n_warmup']
    n_const = cfg['n_const']
    n_print = cfg['n_print'] 
    b_input = cfg['b_input']
    b_update = cfg['b_update']
    
    opt = ScheduledOptim(512, n_warmup, n_const, lr)
    model = opt.initialize(model, fp16=fp16)

    tr_data, cv_dat = datasets
    pool = EpochPool(5)
    epoch_i, _ = load_last_chkpt(model_path, model, opt)
    
    while epoch_i < epochs:
        epoch_i += 1
        print('[ Epoch', epoch_i, ']')
        
        start = time.time()
        tr_loss, tr_accu = train_epoch(model, tr_data, opt, eps, device, b_input, b_update, n_print,
                                       loss_norm=loss_norm, grad_norm=grad_norm)
            
        print('  (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(tr_loss, 100)), accu=100*tr_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        cv_loss, cv_accu = eval_epoch(model, cv_dat, device, b_input)
        print('  (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(cv_loss, 100)), accu=100*cv_accu,
                    elapse=(time.time()-start)/60))

        if math.isnan(cv_loss): break
        model_file = model_path + '/epoch-{}.pt'.format(epoch_i)
        pool.save(cv_loss, model_file, model)
        save_last_chkpt(model_path, epoch_i, model, opt)
