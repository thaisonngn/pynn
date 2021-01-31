# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import math

import torch
from torch.cuda.amp import autocast

from pynn.util.text import token_error_rate

from . import ScheduledOptim, EpochPool, CTCLoss
from . import load_last_chkpt, save_last_chkpt

def compute_ter(preds, targets, blank=0):
    seqs = torch.argmax(preds.detach(), -1).cpu().numpy()
    hypos = []
    for seq in seqs:
        hypo, prev = [], -1
        for pred in seq:
            if pred != prev and pred != blank:
                hypo.append(pred)
                prev = pred
            if pred == blank: prev = -1
        hypos.append(hypo)

    refs = [tg.masked_select(tg.gt(0)).tolist() for tg in targets]
    return token_error_rate(hypos, refs)

def train_epoch(criterion, model, s2s, data, opt, device, b_update, b_sync, n_print,
                weight_noise=False, grad_clip=40., grad_norm=False, fp16=False):
    ''' Epoch operation in training phase'''
    model.train()
    
    total_loss, n_token, total_acc, n_acc = 0., 0, 0., 0
    prints, p_loss, p_token = n_print, 0, 0

    updates, steps = 0, 0
    n_seq = 0

    data_len = len(data)
    loader = data.create_loader()
    opt.zero_grad()
    for batch_i, batch in enumerate(loader):
        # prepare data
        inputs, masks, targets = map(lambda x: x.to(device), batch)
        last = (batch_i == data_len)
        n_seq += targets.size(0)

        try:
            # forward
            with autocast(enabled=fp16):
                # Gaussian weight noise
                if weight_noise: opt.apply_weight_noise()

                attn, masks = s2s.attend(inputs, masks, targets)[1:3]
                attn = attn[:, 0, 1:-1, :].contiguous()
                targets = targets[:, 1:-1].contiguous()
                preds = model(attn.detach(), masks.detach(), targets)

                loss, tgs = criterion(preds, masks, targets)
                if torch.isnan(loss.data):
                    print("    inf loss at %d" % n_seq); continue
            # backward
            opt.backward(loss)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print('    WARNING: ran out of memory on GPU at %d' % n_seq)
                torch.cuda.empty_cache(); continue
            raise err

        updates += tgs; steps += 1
        # update parameters
        if last or (updates >= b_update and b_update > 0) or (steps >= b_sync and b_sync > 0):
            norm = updates if grad_norm else 1
            opt.step_and_update_lr(grad_clip, norm)
            opt.zero_grad()
            updates, steps = 0, 0

        tgs = int(tgs)
        loss_data = float(loss.item())
        total_loss += loss_data; n_token += tgs
        p_loss += loss_data; p_token += tgs
        
        if n_seq > prints and n_print > 0:
            acc = 1 - compute_ter(preds.cpu(), targets.cpu())
            p_loss /= p_token
            print('    Seq: {:6d}, lr: {:.7f}, loss: {:9.4f}, '\
                  'updates: {:6d}, correct: {:.2f}'.format(n_seq, opt.lr, p_loss, opt.steps, acc))
            total_acc += acc; n_acc += 1
            p_loss, p_token = 0., 0
            prints += n_print
    
    total_loss = total_loss / n_token
    total_acc = total_acc / n_acc
    return total_loss, total_acc

def eval_epoch(criterion, model, s2s, data, device, fp16):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss = 0.; n_loss = 0
    total_acc = 0. ; n_acc = 0

    loader = data.create_loader()
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            # prepare data
            inputs, masks, targets = map(lambda x: x.to(device), batch)
            
            # forward
            with autocast(enabled=fp16):
                attn, masks = s2s.attend(inputs, masks, targets)[1:3]
                attn = attn[:, 0, 1:-1, :].contiguous()
                targets = targets[:, 1:-1].contiguous()
                preds = model(attn.detach(), masks.detach(), targets)

                loss, tgs = criterion(preds, masks, targets)

            total_loss += float(loss.item())
            n_loss += int(tgs)
            acc = 1 - compute_ter(preds.cpu(), targets.cpu())
            total_acc += acc; n_acc += 1

    total_loss = total_loss / n_loss
    total_acc = total_acc / n_acc
    return total_loss, total_acc

def train_model(model, s2s, datasets, epochs, device, cfg, fp16=False, dist=False):
    ''' Start training '''
    model_path = cfg['model_path']
    lr = cfg['lr']
    grad_norm = cfg.get('grad_norm', False)
    weight_decay = cfg.get('weight_decay', 0.)
    weight_noise = cfg.get('weight_noise', False)

    n_warmup = cfg.get('n_warmup', 0)
    n_const = cfg.get('n_const', 0)
    n_print = cfg.get('n_print', 1000)
    b_input = cfg.get('b_input', 16000)
    b_sample = cfg.get('b_sample', 64)
    b_update = cfg.get('b_update', 8000)
    b_sync = cfg.get('b_sync', 0)

    n_save = cfg.get('n_save', 5)
    n_print = 0 if dist and device > 0 else n_print
    
    opt = ScheduledOptim(n_warmup, n_const, lr)
    model_opt = opt.initialize(model, device, weight_decay=weight_decay, dist=dist)
    s2s = s2s.to(device)
 
    criterion = CTCLoss()

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
        tr_loss, tr_accu = train_epoch(criterion, model_opt, s2s, tr_data, opt, device, b_update, b_sync,
                                       n_print, weight_noise, grad_norm=grad_norm, fp16=fp16)
        if dist and device > 0: continue
        print('  (Training)   loss: {:8.5f}, accuracy: {:3.3f} %, elapse: {:3.3f} min'.format(
                  tr_loss, 100*tr_accu, (time.time()-start)/60))

        start = time.time()
        cv_loss, cv_accu = eval_epoch(criterion, model_opt, s2s, cv_dat, device, fp16=fp16)
        print('  (Validation) loss: {:8.5f}, accuracy: {:3.3f} %, elapse: {:3.3f} min'.format(
                  cv_loss, 100*cv_accu, (time.time()-start)/60))

        if math.isnan(cv_loss): continue
        model_file = model_path + '/epoch-{}.pt'.format(epoch_i)
        pool.save(cv_loss, model_file, model)
        save_last_chkpt(model_path, epoch_i, model, opt)
