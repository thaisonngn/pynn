# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import math

import torch
from torch.cuda.amp import autocast

from . import EpochPool, load_last_chkpt, save_last_chkpt
from . import ScheduledOptim, cal_ce_loss

def train_epoch(model, data, opt, eps, device, b_update, b_sync, n_print,
        grad_clip=0., grad_norm=False, fp16=False):
    ''' Epoch operation in training phase'''
    model.train()
    
    total_loss, n_token_total, n_token_correct = 0., 0, 0
    prints, p_loss, p_total, p_correct = n_print, 0., 0, 0

    updates, steps = 0, 0
    n_seq = 0

    data_len = len(data)
    loader = data.create_loader()
    opt.zero_grad()
    for batch_i, batch in enumerate(loader):
        # prepare data
        seqs = batch.to(device)
        gold = seqs[:, 1:]
        inputs = seqs[:, :-1]
        last = (batch_i == data_len)
        n_seq += seqs.size(0)

        try:
            # forward
            with autocast(enabled=fp16):
                pred = model(inputs)
                loss, loss_data, n_correct, n_token = cal_ce_loss(pred, gold, eps)
                if torch.isnan(loss.data):
                    print("    inf loss at %d" % n_seq); continue
            # backward
            opt.backward(loss)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print('    WARNING: ran out of memory on GPU at %d' % n_seq)
                torch.cuda.empty_cache(); continue
            raise err

        updates += n_token; steps += 1
        # update parameters
        if last or (updates >= b_update and b_update > 0) or (steps >= b_sync and b_sync > 0):
            norm = updates if grad_norm else 1
            opt.step_and_update_lr(grad_clip, norm)
            opt.zero_grad()
            updates, steps = 0, 0

        # note keeping
        total_loss += loss_data

        n_token_total += n_token;  n_token_correct += n_correct
        p_loss += loss_data; p_total += n_token; p_correct += n_correct
        
        if n_seq > prints and n_print > 0:
            ppl = math.exp(min(p_loss/p_total, 100))
            pred = p_correct * 1. / p_total
            print('    Seq: {:6d}, lr: {:.7f}, ppl: {:9.4f}, '\
                  'updates: {:6d}, correct: {:.2f}'.format(n_seq, opt.lr, ppl, opt.steps, pred))
            prints += n_print
            p_loss = 0.; p_total = 0; p_correct = 0
    
    loss_per_token = total_loss / n_token_total
    accuracy = n_token_correct / n_token_total
    return loss_per_token, accuracy

def eval_epoch(model, data, device, fp16=False):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss = 0
    n_token_total = 0
    n_token_correct = 0

    data_len = len(data)
    loader = data.create_loader()
    with torch.no_grad():
        for batch in loader:
            # prepare data
            seqs = batch.to(device)
            gold = seqs[:, 1:]
            inputs = seqs[:, :-1]

            # forward
            with autocast(enabled=fp16):
                pred = model(inputs)
                loss, loss_data, n_correct, n_token = cal_ce_loss(pred, gold)

            total_loss += loss_data
            non_pad_mask = gold.ne(0)
            n_token = non_pad_mask.sum().item()
            n_token_total += n_token
            n_token_correct += n_correct

    loss_per_token = total_loss / n_token_total
    accuracy = n_token_correct / n_token_total
    return loss_per_token, accuracy
    
def train_model(model, datasets, epochs, device, cfg, fp16=False, dist=False):
    ''' Start training '''
    model_path = cfg['model_path']
    lr = cfg['lr']
    grad_norm = cfg.get('grad_norm', False)
    eps = cfg['label_smooth']
    n_warmup = cfg.get('n_warmup', 0)
    n_const = cfg.get('n_const', 0)
    n_print = cfg.get('n_print', 1000)
    b_input = cfg.get('b_input', 0)
    b_sample = cfg.get('b_sample', 256)
    b_update = cfg.get('b_update', 8000)
    b_sync = cfg.get('b_sync', 0)

    n_save = cfg.get('n_save', 5)
    n_print = 0 if dist and device > 0 else n_print
    
    opt = ScheduledOptim(n_warmup, n_const, lr)
    model_opt = opt.initialize(model, device, dist=dist)
    
    tr_data, cv_dat = datasets
    pool = EpochPool(n_save)
    epoch_i, _ = load_last_chkpt(model_path, model, opt)

    tr_data.initialize(b_input, b_sample)
    cv_dat.initialize(b_input, b_sample)
    while epoch_i < epochs:
        tr_data.set_epoch(epoch_i)
        epoch_i += 1
        if n_print > 0: print('[ Epoch', epoch_i, ']')
        start = time.time()
        tr_loss, tr_accu = train_epoch(model_opt, tr_data, opt, eps, device, b_update, b_sync,
                                       n_print, grad_norm=grad_norm, fp16=fp16)
        if dist and device > 0: continue
        print('  (Training)   ppl: {:8.5f}, accuracy: {:3.3f} %, elapse: {:3.3f} min'.format(
                 math.exp(min(tr_loss, 100)), 100*tr_accu, (time.time()-start)/60))

        start = time.time()
        cv_loss, cv_accu = eval_epoch(model_opt, cv_dat, device, fp16=fp16)
        print('  (Validation) ppl: {:8.5f}, accuracy: {:3.3f} %, elapse: {:3.3f} min'.format(
                 math.exp(min(cv_loss, 100)), 100*cv_accu, (time.time()-start)/60))

        if math.isnan(cv_loss): break
        model_file = model_path + '/epoch-{}.pt'.format(epoch_i)
        pool.save(cv_loss, model_file, model)
        save_last_chkpt(model_path, epoch_i, model, opt)
