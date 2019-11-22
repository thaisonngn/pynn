# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import os
import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pynn.util.decoder import Decoder
from . import change_lr, save_model, load_last_model

def compute_ter(outputs, targets, target_sizes, average=True):
    refs = []
    ofs = 0
    targets = targets.cpu().numpy().tolist()
    for size in target_sizes:
        s = int(size)
        refs.append(targets[ofs:ofs+s])
        ofs += s

    hypos = Decoder.decode(outputs.cpu())
    err = 0
    l = 0
    ter = 0.
    for i in range(len(refs)):
        s = Decoder.score(hypos[i], refs[i])
        err += s[0]
        l += s[1]
        ter += s[2]
    if average: ter /= len(refs)
    return err, l, ter


def train_model(model, datasets, epochs, device, cfg, batch_size=16, alpha=0.00,
        start_lr=0.00005, momentum=0.9, new_bob=(20, 12, 0.8), avg=False, gpu=True):
    print("train_model: start_lr=%f, mementum=%f, alpha=%f" % (start_lr, momentum, alpha))
    batch_sizes = [(1000, 16), (0, 16)]
    epochs, init_lr_epoch, lr_decay = new_bob
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=momentum)
    reduction = "elementwise_mean" if avg else "sum"
    criterion = nn.CTCLoss(reduction=reduction)
    inf = float("inf")
    
    model_path = 'model'
    epoch, best_acc, best_model = load_last_model(model, epochs, model_path)
    
    while epoch < epochs:
        epoch += 1
        since = time.time()

        lr = start_lr 
        if epoch >= init_lr_epoch:
            lr *= lr_decay**(epoch - init_lr_epoch)
        change_lr(optimizer, lr)
        print('Epoch {} using Learning Rate: {}'.format(epoch, lr))

        # Iterate over train data.
        for data, data_type in datasets:
            epoch_loss = 0.
            epoch_err = 0
            epoch_ref = 0
            seq = 0
            #batch_size = batch_sizes[0][1]

            model.train(data_type=='train')
            data.initialize()
            while data.available():
                batch = data.next_batch(batch_size)
                inputs, masks, targets = map(lambda x: x.to(device), batch)
                
                # forward
                #print(inputs.size())
                outputs, masks = model(inputs, masks)
                #print(outputs.size())
                # backward
                pred = outputs.transpose(0, 1) # seq x batch x dim
                #print(pred.size())
                input_sizes = masks.sum(-1)
                #print(targets)
                target_sizes = targets.gt(0).sum(-1)
                #print(target_sizes)       
                seq += 1
                
                loss = criterion(F.log_softmax(pred, dim=-1), targets, input_sizes, target_sizes)
        
                loss = loss if avg else loss / outputs.size(0)
                epoch_loss += loss.data

                if data_type == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 40)
                    optimizer.step()

                if data_type == 'train' and seq % 100 == 0:
                    #ter = compute_ter(outputs, targets, target_sizes)[2]
                    ter = 0.5
                    print("  sequence: %-8d batch-size: %-6d loss: %-7.3f ter: %4.2f" %
                            (seq, batch_size, loss.data, ter))

                if data_type == 'eval':
                    outputs = outputs.transpose(0, 1)
                    err, ref, _ = compute_ter(outputs, targets, target_sizes, False)
                    epoch_err += err
                    epoch_ref += ref
                
                #batch_size = detect_size(input_sizes[0], batch_sizes)

            epoch_loss /= seq
            epoch_ter =  (epoch_err+1.) / (epoch_ref+1)
            epoch_acc = 1. - epoch_ter
            print("  Dataset: %-8s AvgLoss: %7.2f TER: %6.4f" % (data_type, epoch_loss, epoch_ter))

        if epoch_acc > best_acc:
            best_model = copy.deepcopy(model.state_dict())
            best_acc = epoch_acc
            save_model(model, epoch, model_path, epoch_acc)
        elif epoch_acc < best_acc*0.98:
            model.load_state_dict(best_model)
            print('  Low accuracy, skipped the epoch!')

        time_elapsed = time.time() - since
        print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))

    return model
