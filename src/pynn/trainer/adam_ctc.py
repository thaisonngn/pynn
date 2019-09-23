# a trainer class

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

from pynn.util.decoder import Decoder

from . import ScheduledOptim, EpochPool
from . import load_last_chkpt, save_last_chkpt

def compute_ter(outputs, targets, target_sizes, average=True):
    refs = []
    targets = targets.numpy().tolist()
    for idx in range(len(target_sizes)):
        size = int(target_sizes[idx])
        refs.append(targets[idx][0:size])

    hypos = Decoder.decode(outputs)
    ter = 0.
    for i in range(len(refs)):
        ter += Decoder.score(hypos[i], refs[i])[2]
    if average: ter /= len(refs)
    return ter
    
def train_epoch(criterion, model, data, opt, eps, device, batch_input, batch_update, n_print, grad_clip=40.):
    ''' Epoch operation in training phase'''
    model.train()
    
    total_loss = 0.; n_loss = 0
    total_acc = 0. ; n_acc = 0
    prints = n_print

    updates = 0
    n_seq = 0
    data.initialize()
    opt.zero_grad()
    while data.available():
        # prepare data
        batch = data.next(batch_input)
        seqs, tgs, last = batch[-3:]
        batch = batch[:-3]
        inputs, masks, targets = map(lambda x: x.to(device), batch)
        n_seq += seqs

        # forward
        outputs, masks = model(inputs, masks)
        # backward
        input_sizes = masks.sum(-1)
        target_sizes = targets.gt(0).sum(-1)
        pred = F.log_softmax(outputs.transpose(0, 1), dim=-1) # seq x batch x dim
        loss = criterion(pred, targets, input_sizes, target_sizes)
        loss.backward()

        updates += tgs
        # update parameters
        if last or updates >= batch_update:
            if grad_clip > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step_and_update_lr()
            opt.zero_grad()
            updates = 0

        # note keeping
        total_loss += loss.data
        n_loss += 1
        
        if n_seq > prints:
            outputs, targets = outputs.cpu(), targets.cpu()
            acc = 1 - compute_ter(outputs, targets, target_sizes)
            print('    Seq: {:6d}, lr: {:.7f}, loss: {:9.4f}, '\
                       'updates: {:6d}, correct: {:.2f}'.format(n_seq, opt.lr, loss.data, opt.steps, acc))
            total_acc += acc; n_acc += 1
            prints += n_print
    
    total_loss = total_loss / n_loss
    total_acc = total_acc / n_acc
    return total_loss, total_acc

def eval_epoch(criterion, model, data, device, batch_input):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0.; n_loss = 0
    total_acc = 0. ; n_acc = 0
    
    with torch.no_grad():
        data.initialize()
        while data.available():
            # prepare data
            batch = data.next(batch_input)
            batch = batch[:-3]
            inputs, masks, targets = map(lambda x: x.to(device), batch)
            
            # forward
            outputs, masks = model(inputs, masks)
            # backward
            input_sizes = masks.sum(-1)
            target_sizes = targets.gt(0).sum(-1)
            pred = F.log_softmax(outputs.transpose(0, 1), dim=-1)
            loss = criterion(pred, targets, input_sizes, target_sizes)

            total_loss += loss.data
            n_loss += 1

            outputs, targets = outputs.cpu(), targets.cpu()
            acc = 1 - compute_ter(outputs, targets, target_sizes)
            total_acc += acc; n_acc += 1

    total_loss = total_loss / n_loss
    total_acc = total_acc / n_acc
    return total_loss, total_acc
    
def train_model(model, datasets, epochs, device, cfg):
    ''' Start training '''

    model_path = cfg['model_path']
    lr = cfg['lr']
    eps = cfg['smooth']

    n_warmup = cfg['n_warmup']
    n_print = cfg['n_print'] 
    b_input = cfg['b_input']
    b_update = cfg['b_update']
    
    opt = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09), 512, n_warmup, lr)

    criterion = nn.CTCLoss(reduction='sum')

    tr_data, cv_dat = datasets
    pool = EpochPool(5)
    epoch_i, _ = load_last_chkpt(model_path, model, opt)
    
    while epoch_i < epochs:
        epoch_i += 1
        print('[ Epoch', epoch_i, ']')
        
        start = time.time()
        tr_loss, tr_accu = train_epoch(criterion, model, tr_data, opt, eps, device, b_input, b_update, n_print)
            
        print('  (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(loss=tr_loss, accu=100*tr_accu, elapse=(time.time()-start)/60))

        start = time.time()
        cv_loss, cv_accu = eval_epoch(criterion, model, cv_dat, device, b_input)
        print('  (Validation) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(loss=cv_loss, accu=100*cv_accu, elapse=(time.time()-start)/60))

        if math.isnan(cv_loss): break
        model_file = model_path + '/epoch-{}.pt'.format(epoch_i)
        pool.save(cv_loss, model_file, model)
        save_last_chkpt(model_path, epoch_i, model, opt)
