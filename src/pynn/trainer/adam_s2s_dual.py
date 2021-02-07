# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import math

import torch
from torch.cuda.amp import autocast

from . import EpochPool, ScheduledOptim
from . import cal_ce_loss, load_last_chkpt, save_last_chkpt

def train_epoch(model, data, opt, eps, device, b_update, b_sync, n_print,
                alpha=0.5, teacher_force=1., weight_noise=False, grad_clip=0., grad_norm=False, fp16=False):
    ''' Epoch operation in training phase'''
    model.train()
    
    prints, total_loss, n_token_total, n_token_correct = n_print, 0., 0, 0
    p_loss_pre, p_token_pre, p_correct_pre = 0., 0, 0
    p_loss_pos, p_token_pos, p_correct_pos = 0., 0, 0

    updates, steps = 0, 0
    n_seq = 0

    data_len = len(data)
    loader = data.create_loader()
    opt.zero_grad()
    for batch_i, batch in enumerate(loader):
        # prepare data
        src_seq, src_mask, tgt_pre, tgt_pos = map(lambda x: x.to(device), batch)
        last = (batch_i == data_len)
        n_seq += src_seq.size(0)

        try:
            # forward
            with autocast(enabled=fp16):
                # Gaussian weight noise
                if weight_noise: opt.apply_weight_noise()
                gold_pre, tgt_pre = tgt_pre[:, 1:], tgt_pre[:, :-1]
                gold_pos, tgt_pos = tgt_pos[:, 1:], tgt_pos[:, :-1]
                pred_pre, pred_pos = model(src_seq, src_mask, tgt_pre, tgt_pos)[0:2]
                loss_pre, loss_data_pre, n_correct_pre, n_token_pre = cal_ce_loss(pred_pre, gold_pre)[0]
                loss_pos, loss_data_pos, n_correct_pos, n_token_pos = cal_ce_loss(pred_pos, gold_pos, eps)
                loss = loss_pre * alpha + loss_pos * (1 - alpha)
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
        p_loss_pre += loss_data_pre; p_token_pre += n_token_pre; p_correct_pre += n_correct_pre
        p_loss_pos += loss_data_pos; p_token_pos += n_token_pos; p_correct_pos += n_correct_pos
        total_loss += loss_data_pos; n_token_total += n_token_pos;  n_token_correct += n_correct_pos
        
        if n_seq > prints and n_print > 0:
            ppl_pre, pred_pre = math.exp(min(p_loss_pre/p_token_pre, 100)), p_correct_pre * 1. / p_token_pre
            ppl_pos, pred_pos = math.exp(min(p_loss_pos/p_token_pos, 100)), p_correct_pos * 1. / p_token_pos
            print('    Seq: {:6d}, lr: {:.7f}, ppl: {:9.4f}, {:9.4f} '\
                  'updates: {:6d}, correct: {:.2f}, {:.2f}'.format(
                  n_seq, opt.lr, ppl_pre, ppl_pos, opt.steps, pred_pre, pred_pos), flush=True)
            prints += n_print
            p_loss_pre, p_token_pre, p_correct_pre = 0., 0, 0
            p_loss_pos, p_token_pos, p_correct_pos = 0., 0, 0
    
    loss_per_token = total_loss / n_token_total
    accuracy = n_token_correct / n_token_total
    return loss_per_token, accuracy

def eval_epoch(model, data, device, fp16=False):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss, n_token_total, n_token_correct = 0., 0, 0

    loader = data.create_loader()
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            # prepare data
            src_seq, src_mask, tgt_pre, tgt_pos = map(lambda x: x.to(device), batch)

            # forward
            with autocast(enabled=fp16):
                gold_pre, tgt_pre = tgt_pre[:, 1:], tgt_pre[:, :-1]
                gold_pos, tgt_pos = tgt_pos[:, 1:], tgt_pos[:, :-1]
                pred_pre, pred_pos = model(src_seq, src_mask, tgt_pre, tgt_pos)[0:2]
                loss, loss_data, n_correct, n_token = cal_ce_loss(pred_pos, gold_pos)
            total_loss += loss_data
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
    eps = cfg.get('label_smooth', 0.)
    teacher_force = cfg.get('teacher_force', 1.)
    weight_decay = cfg.get('weight_decay', 0.)
    weight_noise = cfg.get('weight_noise', False)

    n_warmup = cfg.get('n_warmup', 0)
    n_const = cfg.get('n_const', 0)
    n_print = cfg.get('n_print', 1000)
    b_input = cfg.get('b_input', 20000)
    b_sample = cfg.get('b_sample', 64)
    b_update = cfg.get('b_update', 8000)
    b_sync = cfg.get('b_sync', 0)

    n_save = cfg.get('n_save', 5)
    n_print = 0 if dist and device > 0 else n_print

    opt = ScheduledOptim(n_warmup, n_const, lr)
    model_opt = opt.initialize(model, device, weight_decay=weight_decay, dist=dist)

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
        tr_loss, tr_accu = train_epoch(model_opt, tr_data, opt, eps, device, b_update, b_sync, n_print,
                                       teacher_force, weight_noise, grad_norm=grad_norm, fp16=fp16)
        if dist and device > 0: continue
        print('  (Training)   ppl: {:8.5f}, accuracy: {:3.3f} %, elapse: {:3.3f} min'.format(
                 math.exp(min(tr_loss, 100)), 100*tr_accu, (time.time()-start)/60))

        start = time.time()
        cv_loss, cv_accu = eval_epoch(model_opt, cv_dat, device, fp16=fp16)
        print('  (Validation) ppl: {:8.5f}, accuracy: {:3.3f} %, elapse: {:3.3f} min'.format(
                 math.exp(min(cv_loss, 100)), 100*cv_accu, (time.time()-start)/60))

        if math.isnan(cv_loss): continue
        model_file = model_path + '/epoch-{}.pt'.format(epoch_i)
        pool.save(cv_loss, model_file, model)
        save_last_chkpt(model_path, epoch_i, model, opt)
