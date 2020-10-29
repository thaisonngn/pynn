# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, d_model, w_steps, c_steps, lr=1.0):
        self.amp = None
        self.w_steps = w_steps
        self.c_steps = c_steps
        self.steps = 0
        self.lr = lr
        if self.w_steps == 0:
            self.init_lr = lr
        else:
            self.init_lr = d_model**(-0.5) * self.lr

    def initialize(self, model, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, fp16=False):
        self.params = filter(lambda x: x.requires_grad, model.parameters())
        self.optim = optim.Adam(self.params, betas=betas, eps=eps, weight_decay=weight_decay)
        if fp16:
            from apex import amp
            model, self.optim = amp.initialize(model, self.optim, opt_level="O2")
            self.amp = amp
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model) #enabling data parallelism
        return model

    def backward(self, loss):
        if self.amp:
            with self.amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step_and_update_lr(self, norm=1.):
        "Step with the inner optimizer"
        self._update_learning_rate()

        for p in self.params:
            p.grad.data.div_(norm)
        self.optim.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optim.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.steps += 1
        if self.w_steps == 0:
            scale = 1.0
        elif self.steps <= self.w_steps:
            scale = self.steps*(self.w_steps**-1.5)
        elif self.c_steps == 0:
            scale = self.steps**-0.5
        else:
            n = (self.steps-self.w_steps) // self.c_steps
            n = min(n, 10)
            scale = (self.steps**-0.5) * (0.8**n)

        self.lr = self.init_lr * scale
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr

    def state_dict(self):
        state_dict = self.optim.state_dict()
        state_dict['steps'] = self.steps
        if self.amp:
            state_dict['amp'] = self.amp.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.steps = state_dict.pop('steps', 0)
        amp_state = state_dict.pop('amp', None)
        if self.amp and amp_state:
            self.amp.load_state_dict(amp_state)
        self.optim.load_state_dict(state_dict)

class EpochPool(object):
    def __init__(self, save=5):
        self.saves = [(100., '') for epoch in range(save)]

    def save(self, err, path, model):
        highest_err = self.saves[-1]
        if highest_err[0] < err:
            return
        if os.path.isfile(highest_err[1]):
            os.remove(highest_err[1])

        self.saves[-1] = (err, path)
        torch.save(model.state_dict(), path)
        self.saves.sort(key=lambda e : e[0])

def cal_ce_loss(pred, gold, eps=0.0):
    lprobs = F.log_softmax(pred, dim=-1).view(-1, pred.size(-1))
    gtruth = gold.contiguous().view(-1)  # batch * time
    
    non_pad_mask = gtruth.ne(0)
    nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
    nll_loss = nll_loss.sum()
    loss_data = nll_loss.data.item()
    
    if eps > 0.:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        smooth_loss = smooth_loss.sum()
        eps_i =  eps / (lprobs.size(-1) - 2)
        loss = (1. - eps) * nll_loss + eps_i * smooth_loss
    else:
        loss = nll_loss
        
    lprobs = lprobs.argmax(dim=-1)
    n_correct = lprobs.eq(gtruth)[non_pad_mask]
    n_correct = n_correct.sum().item()
    n_total = non_pad_mask.sum().item()
    
    return loss, loss_data, n_correct, n_total

def freeze_model(layer):
    for param in layer.parameters():
        param.requires_grad = False
    return layer
    
def change_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
    
def save_last_chkpt(path, epoch, model, optimizer=None, state=None):
    opt_state = '' if optimizer is None else optimizer.state_dict()
    dic = {'model_state': model.state_dict(), 'epoch': epoch,
           'optimizer_state': opt_state, 'epoch_state': state}
    chkp_file = path + '/last-epoch.chkpt'
    torch.save(dic, chkp_file)

def load_last_chkpt(path, model, optimizer=None, fine_tuning=False, device: str = 'cpu'):
    if fine_tuning and os.path.isfile(path):
        chkp_file = path
    else:
        chkp_file = path + '/last-epoch.chkpt'
        if not os.path.isfile(chkp_file):
            return 0, None
    dic = torch.load(chkp_file, map_location=device)
    if fine_tuning:
        filename: str = os.path.basename(chkp_file)
        state_dict = {}
        if filename.startswith('epoch-avg') and filename.endswith('.dic'):
            state_dict = dic['state']
        elif filename.startswith('epoch-') and filename.endswith('.pt'):
            state_dict = dic
        del state_dict['decoder.project.weight']
        del state_dict['decoder.project.bias']
        del state_dict['decoder.emb.weight']
        if 'epoch' not in dic:
            dic['epoch'] = None
        if 'epoch_state' not in dic:
            dic['epoch_state'] = None
    else:
        state_dict = dic['model_state']
    model.load_state_dict(state_dict, strict=not fine_tuning)
    if optimizer is not None:
        optimizer.load_state_dict(dic['optimizer_state'])
    return dic['epoch'], dic['epoch_state']
