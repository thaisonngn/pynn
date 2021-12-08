# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, w_steps, c_steps, lr=1.0):
        self.amp = None
        self.w_steps = w_steps
        self.c_steps = c_steps
        self.steps = 0
        self.init_lr = lr if w_steps == 0 else lr / w_steps**-0.5
        self.lr = 0.

    def initialize(self, model, device, params=None,
            betas=(0.9, 0.98), eps=1e-09, weight_decay=0, dist=False):
        model = model.to(device)
        self.params = filter(lambda x: x.requires_grad, model.parameters()) if params is None else params
        self.optim = optim.Adam(self.params, betas=betas, eps=eps, weight_decay=weight_decay)
        self.scaler = GradScaler()
        if dist:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, find_unused_parameters=True, device_ids=[device])
        return model

    def apply_weight_noise(self):
        with torch.no_grad():
            for p in self.params: p.add_(torch.normal(0,  0.075, param.size())) 

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step_and_update_lr(self, grad_clip=0., grad_norm=1.):
        "Step with the inner optimizer"
        self._update_learning_rate()

        if grad_clip > 0.:
            nn.utils.clip_grad_norm_(self.params, grad_clip)

        if grad_norm > 1.:
            for p in self.params: p.grad.data.div_(grad_norm)
        
        self.scaler.step(self.optim)
        self.scaler.update()

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
        state_dict['scaler'] = self.scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.steps = state_dict.pop('steps', 0)
        scaler_state = state_dict.pop('scale', None)
        if scaler_state:
            self.scaler.load_state_dict(scaler_state)
        self.optim.load_state_dict(state_dict)

class EpochPool(object):
    def __init__(self, save=5):
        self.saves = [(10000., '') for epoch in range(save)]

    def save(self, err, path, model):
        highest_err = self.saves[-1]
        if highest_err[0] < err:
            return
        if os.path.isfile(highest_err[1]):
            os.remove(highest_err[1])

        self.saves[-1] = (err, path)
        torch.save(model.state_dict(), path)
        self.saves.sort(key=lambda e : e[0])

class CTCLoss(nn.Module):
    def __init__(self, mean=False):
        super().__init__()

        reduction = 'mean' if mean else 'sum'
        self.loss = nn.CTCLoss(reduction=reduction, zero_infinity=True)
        self.activate = lambda x: F.log_softmax(x.float(), dim=-1)

    def forward(self, logits, masks, targets):
        input_sizes = masks.sum(-1)
        target_sizes = targets.gt(0).sum(-1)

        logits = self.activate(logits.transpose(0, 1)) # seq x batch x dim
        loss = self.loss(logits, targets, input_sizes, target_sizes)
        tgs = target_sizes.sum()

        return loss, tgs

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

def cal_kl_loss(p, q, mask):
    mask = mask.view(-1)
    p_lp = F.log_softmax(p, dim=-1).view(-1, p.size(-1))[mask]
    q_pb = F.softmax(q, dim=-1).view(-1, q.size(-1))[mask]
    kl_loss = q_pb * p_lp
    kl_loss = -kl_loss.sum()
    loss_data = kl_loss.data.item()

    return kl_loss, loss_data

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

def load_last_chkpt(path, model, optimizer=None):
    chkp_file = path + '/last-epoch.chkpt'
    if not os.path.isfile(chkp_file):
        return 0, None
    dic = torch.load(chkp_file, map_location='cpu')
    model.load_state_dict(dic['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(dic['optimizer_state'])
    return dic['epoch'], dic['epoch_state']
