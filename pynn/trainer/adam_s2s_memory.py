# Copyright 2019 Thai-Son Nguyen, Christian Huber
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import math, random

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from . import EpochPool, ScheduledOptim
from . import cal_ce_loss, load_last_chkpt, save_last_chkpt

from pynn.io.memory_dataset import MemoryDataset

def pred_to_loss(pred, gold, non_pad_mask, eps=0.0, no_softmax=False):
    lprobs = F.log_softmax(pred, dim=-1).view(-1, pred.size(-1)) if not no_softmax \
             else torch.log(pred).view(-1, pred.size(-1))
    gtruth = gold.contiguous().view(-1)  # batch * time
    non_pad_mask = non_pad_mask.view(-1)

    nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
    nll_loss = nll_loss.sum()

    if eps > 0.:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        smooth_loss = smooth_loss.sum()
        eps_i = eps / (lprobs.size(-1) - 2)
        loss = (1. - eps) * nll_loss + eps_i * smooth_loss
    else:
        loss = nll_loss

    lprobs = lprobs.argmax(dim=-1)
    n_correct = lprobs.eq(gtruth)[non_pad_mask]
    n_correct = n_correct.sum().item()

    return loss, n_correct

def cal_loss_trainer(pred, mem_attn_outs, gold, label_mem, eps=0.0):
    non_pad_mask = gold.ne(0)
    mask_memory = label_mem.gt(0) & non_pad_mask
    anz_memory = mask_memory.sum().item()
    mask_no_memory = label_mem.eq(0) & non_pad_mask
    anz_no_memory = mask_no_memory.sum().item()

    # calc classification loss for the tokens
    loss_tok_pred_no_memory, n_correct_no_memory = pred_to_loss(pred, gold, mask_no_memory, eps=eps, no_softmax=True)
    loss_tok_pred_memory, n_correct_memory = pred_to_loss(pred, gold, mask_memory, eps=eps, no_softmax=True)

    # mem_attn_outs: (attnGate,attn)
    mem_attn_outs = [(*pred_to_loss(x, label_mem, mask_no_memory, eps=eps),
                      *pred_to_loss(x, label_mem, mask_memory, eps=eps))
                     for x in mem_attn_outs]

    loss_gate_no_memory = torch.stack([x[0] for x in mem_attn_outs]).mean()
    loss_gate_memory = torch.stack([x[2] for x in mem_attn_outs]).mean()

    loss = loss_tok_pred_no_memory/anz_no_memory*(anz_memory+anz_no_memory) + \
           loss_tok_pred_memory/anz_memory*(anz_memory+anz_no_memory) + \
           loss_gate_no_memory / anz_no_memory * (anz_memory + anz_no_memory) + \
           loss_gate_memory / anz_memory * (anz_memory + anz_no_memory)

    stats = torch.tensor([[y.cpu().item() if not type(y) == int else y for y in x] for x in mem_attn_outs])  # n_dec_layer x 4
    stats2 = torch.as_tensor([anz_no_memory, anz_memory, anz_no_memory+anz_memory,
                              loss_tok_pred_no_memory, n_correct_no_memory,
                              loss_tok_pred_memory, n_correct_memory])
    return loss, stats, stats2

def ppl_acc(stats,stats2,id,id2,flip=1):
    ppl = 0
    acc = 0
    for i in id:
        i = list(i)
        tmp = stats[i[0]]
        for j in range(1, len(i)):
            tmp = tmp[i[j]]
        ppl += tmp
        i[-1] += 1
        tmp = stats[i[0]]
        for j in range(1, len(i)):
            tmp = tmp[i[j]]
        acc += tmp
    return math.exp(min(ppl / stats2[id2], 100)), flip * 100 * acc / stats2[id2]

def printStats(stats, stats2, n_seq=-1, lr=-1, steps=-1, time=-1, train=None):
    if n_seq==-1:
        if train:
            print('  (Training)   elapse: %3.3f min'%time)
        else:
            print('  (Validation) elapse: %3.3f min'%time)
    else:
        print('  Seq: {:6d}, lr: {:.7f}, updates: {:6d}, more stats:'.format(n_seq,lr,steps))

    print("    Token pred: No_mem: ppl:%9.4f,acc:%5.2f, Mem: ppl:%9.4f,acc:%5.2f, All: ppl:%9.4f,acc:%5.2f" %
          (*ppl_acc(stats2,stats2,[(3,)],0),*ppl_acc(stats2,stats2,[(5,)],1),*ppl_acc(stats2,stats2,[(3,),(5,)],2)))

    for y in range(stats.shape[0]):
        print("    After layer %1d: Gate: No_mem: ppl:%7.4f,acc:%6.2f, Mem: ppl:%7.4f,acc:%6.2f, All: ppl:%7.4f,acc:%6.2f"
              %(y+1,*ppl_acc(stats,stats2,[(y,0)],0),
                    *ppl_acc(stats,stats2,[(y,2)],1),
                    *ppl_acc(stats,stats2,[(y,0),(y,2)],2)))

    return (stats2[3]+stats2[5])/stats2[2] # loss token prediction

def train_epoch(model, data, opt, eps, device, args, b_update, b_sync, n_print,
                teacher_force=1., weight_noise=False, grad_clip=0., grad_norm=False, fp16=False):
    ''' Epoch operation in training phase'''
    model.train()
    prints = n_print

    stats2 = torch.zeros(7)
    p_stats2 = torch.zeros(7)
    stats = torch.zeros((args.n_dec_mem, 4))
    p_stats = torch.zeros((args.n_dec_mem, 4))

    updates, steps = 0, 0
    n_seq = 0

    OOM = 0
    bs_factor = 1

    data_len = len(data)

    data.shuffle()
    loader = data.create_loader()

    opt.zero_grad()
    for batch_i, batch in enumerate(loader):
        if n_seq>args.n_seq_max_epoch:
            break
        if OOM>0:
            torch.cuda.empty_cache()
        if OOM>=10:
            bs_factor *= 0.9
            if bs_factor<1/1000:
                print("To small batch size, breaking epoch.")
                break
            print("Reduced batch size factor to",bs_factor)
            OOM = 0

        # prepare data
        src_seq, src_mask, tgt_seq, tgt_ids_mem, label_mem = map(lambda x: x.to(device), batch)

        if bs_factor<1:
            anz = max(1,int(bs_factor * src_seq.shape[0]))
            ids = random.sample(range(src_seq.shape[0]), anz)
            src_seq = src_seq[ids]
            src_mask = src_mask[ids]
            tgt_seq = tgt_seq[ids]
            label_mem = label_mem[ids]

        gold = tgt_seq[:, 1:]
        tgt_seq = tgt_seq[:, :-1]
        last = (batch_i == data_len)
        n_seq += tgt_seq.size(0)

        try:
            # forward
            with autocast(enabled=fp16):
                # Gaussian weight noise
                if weight_noise: opt.apply_weight_noise()

                # teacher forcing or sampling
                sampling = teacher_force < 1.
                if sampling:
                    pred, _, enc_out, _ = model(src_seq, src_mask, tgt_seq, tgt_ids_mem, label_mem, gold)
                    pred = torch.argmax(pred, dim=-1).detach()
                    sample = pred.clone().bernoulli_(1. - teacher_force)
                    pred = pred * sample
                    pred = torch.cat((tgt_seq[:, :1], pred[:, :-1]), dim=1)
                    pred = pred * tgt_seq.gt(2).type(pred.dtype)
                    tgt_seq = tgt_seq * pred.eq(0).type(pred.dtype) + pred
                else:
                    enc_out = None

                pred, mem_attn_outs, _, _ = model(src_seq, src_mask, tgt_seq, tgt_ids_mem, label_mem, gold, encoding=not sampling, enc_out=enc_out)
                pred = pred.view(-1, pred.size(2))
                loss, stats_, stats2_ = cal_loss_trainer(pred, mem_attn_outs, gold, label_mem, eps=eps)
                if torch.isnan(loss.data):
                    print("    inf loss at %d" % n_seq); continue
            # backward
            opt.backward(loss)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print('    WARNING: ran out of memory on GPU at %d' % n_seq)
                OOM += 1
                continue
            raise err
        OOM = 0

        n_token = stats2_[2]
        updates += n_token; steps += 1
        # update parameters
        if last or (updates >= b_update and b_update > 0) or (steps >= b_sync and b_sync > 0):
            norm = updates if grad_norm else 1
            opt.step_and_update_lr(grad_clip, norm)
            opt.zero_grad()
            updates, steps = 0, 0

        # note keeping
        stats += stats_
        p_stats += stats_
        stats2 += stats2_
        p_stats2 += stats2_

        if n_seq > prints:
            printStats(p_stats, p_stats2, n_seq, opt.lr, opt.steps)
            prints += n_print
            p_stats = torch.zeros((args.n_dec_mem, 4))
            p_stats2 = torch.zeros(7)

    return stats, stats2

def eval_epoch(model, data, device, args, fp16=False):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    stats = torch.zeros((args.n_dec_mem, 4))
    stats2 = torch.zeros(7)

    loader = data.create_loader()
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            if batch_i<10:
                continue

            # prepare data
            src_seq, src_mask, tgt_seq, tgt_ids_mem, label_mem = map(lambda x: x.to(device), batch)

            """print("Trainer")
            id = 0
            src_seq = src_seq[id:id+1]
            src_mask = src_mask[id:id+1]
            tgt_seq = tgt_seq[id:id+1]
            print(tgt_seq[0])
            label_mem = label_mem[id:id+1]
            print(label_mem[0])
            ids = set()
            for x in label_mem[0]:
                if not x==0:
                    ids.add(int(x))
            for i in ids:
                print(tgt_ids_mem[i-1])"""

            gold = tgt_seq[:, 1:]
            tgt_seq = tgt_seq[:, :-1]

            # forward
            with autocast(enabled=fp16):
                pred, mem_attn_outs, _, _ = model(src_seq, src_mask, tgt_seq, tgt_ids_mem, label_mem, gold)
                loss, stats_, stats2_ = cal_loss_trainer(pred, mem_attn_outs, gold, label_mem)

            stats += stats_
            stats2 += stats2_

        return stats, stats2

def freeze(model, unfreeze=False):
    for p in model.parameters():
        p.requires_grad = unfreeze

def train_model(model, datasets, epochs, device, args, fp16=False, dist=False):
    ''' Start training '''
    model_path = args.model_path
    n_print = 0 if dist and device > 0 else args.n_print

    if args.freeze_baseline and args.pretrained_model!="None":
        freeze(model.encoder)
        freeze(model.decoder)
        print("Baseline frozen")

    opt = ScheduledOptim(args.n_warmup, args.n_const, args.lr)
    model_opt = opt.initialize(model, device, weight_decay=args.weight_decay, dist=dist)

    tr_data, cv_dat = datasets
    pool = EpochPool(args.n_save)
    epoch_i, _ = load_last_chkpt(model_path, model, opt)
    if epoch_i == 0 and args.pretrained_model!="None":
        model.load_state_dict(torch.load(args.pretrained_model)['state'],strict=False)
        print("Pretrained model loaded")

    tr_data.initialize(args.b_input, args.b_sample)
    tr_data = MemoryDataset(tr_data, args)
    tr_data.shuffle()

    epoch_i_init = epoch_i
    while epoch_i < epochs:
        tr_data.set_epoch(epoch_i)
        epoch_i += 1 
        if n_print > 0: print('[ Epoch', epoch_i, ']')
        start = time.time()
        stats, stats2 = train_epoch(model_opt, tr_data, opt, args.label_smooth, device, args, args.b_update, args.b_sync, n_print,
                                       args.teacher_force, args.weight_noise, grad_norm=args.grad_norm, fp16=fp16)
        if dist and device > 0: continue
        printStats(stats, stats2, time=(time.time() - start) / 60, train=True)

        if epoch_i==epoch_i_init+1:
            cv_dat.initialize(args.b_input, args.b_sample)
            cv_dat = MemoryDataset(cv_dat, args, validation=True)

        start = time.time()
        stats, stats2 = eval_epoch(model_opt, cv_dat, device, args, fp16=fp16)
        cv_loss = printStats(stats, stats2, time=(time.time() - start) / 60, train=False)

        if math.isnan(cv_loss): continue
        model_file = model_path + '/epoch-{}.pt'.format(epoch_i)
        pool.save(cv_loss, model_file, model)
        save_last_chkpt(model_path, epoch_i, model, opt)
