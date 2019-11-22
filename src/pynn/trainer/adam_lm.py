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

def train_epoch(model, data, opt, eps, device, batch_input, batch_update, n_print,
                loss_norm=False, grad_norm=True, grad_clip=0.):
    ''' Epoch operation in training phase'''
    model.train()
    
    total_loss = 0.; n_word_total = 0; n_word_correct = 0
    prints = n_print
    p_loss = 0.; p_total = 0; p_correct = 0

    updates = 0
    n_seq = 0

    data.initialize()
    opt.zero_grad()
    while data.available():
        # prepare data
        seqs, nseq, tgs, last = data.next(batch_input)

        seqs = seqs.to(device)
        gold = seqs[:, 1:]
        inputs = seqs[:, :-1]
        n_seq += nseq

        try:
            # forward
            pred = model(inputs)
            # backward
            loss, loss_data, n_correct = cal_ce_loss(pred, gold, eps)
            if torch.isnan(loss.data):
                print("    inf loss at %d" % n_seq); continue
            if loss_norm: loss = loss.div(tgs)
            opt.backward(loss)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print('    WARNING: ran out of memory on GPU at %d' % n_seq)
                torch.cuda.empty_cache(); continue
            raise err

        updates += tgs
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

        n_word = tgs
        n_word_total += n_word;  n_word_correct += n_correct
        p_loss += loss_data; p_total += n_word; p_correct += n_correct
        
        if n_seq > prints:
            ppl = math.exp(min(p_loss/p_total, 100))
            pred = p_correct * 1. / p_total
            print('    Seq: {:6d}, lr: {:.7f}, ppl: {:9.4f}, '\
                    'updates: {:6d}, correct: {:.2f}'.format(n_seq, opt.lr, ppl, opt.steps, pred))
            prints += n_print
            p_loss = 0.; p_total = 0; p_correct = 0
    
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
            seqs, nseq, tgs, last = data.next(batch_input)

            seqs = seqs.to(device)
            gold = seqs[:, 1:]
            inputs = seqs[:, :-1]

            # forward
            pred = model(inputs)
            loss, loss_data, n_correct = cal_ce_loss(pred, gold)

            # note keeping
            total_loss += loss_data

            non_pad_mask = gold.ne(0)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
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
