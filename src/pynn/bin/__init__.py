# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

from pynn.io.audio_seq import SpectroDataset
from pynn.io.text_seq import TextSeqDataset, TextPairDataset
from pynn.trainer.adam_s2s import train_model as train_s2s
from pynn.trainer.adam_ctc import train_model as train_ctc
from pynn.trainer.adam_lm import train_model as train_lm
from pynn.trainer.adam_hybrid import train_model as train_hybrid
from pynn.trainer.adam_s2s_kl import train_model as distill_s2s
from pynn.trainer.adam_enc import train_model as train_encoder

def print_model(model):
    model_size = sum(p.numel() for p in model.parameters()) / 1000000.
    print('Model size: %.2fM' % model_size)
    
def train_s2s_model(model, args, device, n_device=1):
    dist, verbose = n_device > 1, device == 0
    tr_data = SpectroDataset(args.train_scp, args.train_target, downsample=args.downsample,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             spec_drop=args.spec_drop, spec_bar=args.spec_bar, spec_ratio=args.spec_ratio,
                             time_stretch=args.time_stretch, time_win=args.time_win,
                             threads=2, verbose=verbose)
    cv_data = SpectroDataset(args.valid_scp, args.valid_target, downsample=args.downsample,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             threads=2, verbose=verbose)
    if dist: tr_data.partition(device, n_device)
    n_print = args.n_print // n_device
    b_update = args.b_update // n_device

    cfg = {'model_path': args.model_path, 'lr': args.lr, 'grad_norm': args.grad_norm,
           'weight_decay': args.weight_decay, 'weight_noise': args.weight_noise,
           'label_smooth': args.label_smooth, 'teacher_force': args.teacher_force,
           'n_warmup': args.n_warmup, 'n_const': args.n_const, 'n_save': args.n_save, 'n_print': n_print,
           'b_input': args.b_input, 'b_sample': args.b_sample, 'b_update': b_update, 'b_sync': args.b_sync}
    datasets = (tr_data, cv_data)
    train_s2s(model, datasets, args.n_epoch, device, cfg, fp16=args.fp16, dist=dist)

def train_ctc_model(model, args, device, n_device=1):
    dist, verbose = n_device > 1, device == 0
    tr_data = SpectroDataset(args.train_scp, args.train_target, downsample=args.downsample,
                             sort_src=True, sek=False, mean_sub=args.mean_sub,
                             fp16=args.fp16, preload=args.preload, threads=2, verbose=verbose,
                             spec_drop=args.spec_drop, spec_bar=args.spec_bar, spec_ratio=args.spec_ratio,
                             time_stretch=args.time_stretch, time_win=args.time_win)
    cv_data = SpectroDataset(args.valid_scp, args.valid_target, downsample=args.downsample,
                             sort_src=True, sek=False, mean_sub=args.mean_sub,
                             fp16=args.fp16, preload=args.preload, threads=2, verbose=verbose)
    if dist: tr_data.partition(device, n_device)
    n_print = args.n_print // n_device
    b_update = args.b_update // n_device
    
    cfg = {'model_path': args.model_path, 'lr': args.lr, 'grad_norm': args.grad_norm,
           'weight_decay': args.weight_decay, 'weight_noise': args.weight_noise,
           'n_warmup': args.n_warmup, 'n_const': args.n_const, 'n_save': args.n_save, 'n_print': n_print,
           'b_input': args.b_input, 'b_sample': args.b_sample, 'b_update': b_update, 'b_sync': args.b_sync}
    datasets = (tr_data, cv_data)
    train_ctc(model, datasets, args.n_epoch, device, cfg, fp16=args.fp16, dist=dist)

def train_language_model(model, args, device, n_device=1):
    dist, verbose = n_device > 1, device == 0
    tr_data = TextSeqDataset(args.train_seq, sek=not args.no_sek, threads=2, verbose=verbose)
    cv_data = TextSeqDataset(args.valid_seq, sek=not args.no_sek, threads=2, verbose=verbose)
    if dist: tr_data.partition(device, n_device)
    n_print = args.n_print // n_device
    b_update = args.b_update // n_device
    
    cfg = {'model_path': args.model_path, 'lr': args.lr, 'grad_norm': args.grad_norm,
           'label_smooth': args.label_smooth,
           'n_warmup': args.n_warmup, 'n_const': args.n_const,'n_save': args.n_save, 'n_print': n_print,
           'b_input': args.b_input, 'b_sample': args.b_sample, 'b_update': b_update, 'b_sync': args.b_sync}
    datasets = (tr_data, cv_data)
    train_lm(model, datasets, args.n_epoch, device, cfg, fp16=args.fp16, dist=dist)

def train_hybrid_model(model, args, device, n_device=1):
    dist, verbose = n_device > 1, device == 0
    tr_data = SpectroDataset(args.train_scp, args.train_target, downsample=args.downsample,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             spec_drop=args.spec_drop, spec_bar=args.spec_bar, spec_ratio=args.spec_ratio,
                             time_stretch=args.time_stretch, time_win=args.time_win,
                             threads=2, verbose=verbose)
    cv_data = SpectroDataset(args.valid_scp, args.valid_target, downsample=args.downsample,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             threads=2, verbose=verbose)
    if dist: tr_data.partition(device, n_device)
    n_print = args.n_print // n_device
    b_update = args.b_update // n_device
    
    cfg = {'model_path': args.model_path, 'lr': args.lr, 'grad_norm': args.grad_norm,
           'weight_decay': args.weight_decay, 'weight_noise': args.weight_noise,
           'label_smooth': args.label_smooth, 'teacher_force': args.teacher_force,
           'n_warmup': args.n_warmup, 'n_const': args.n_const, 'n_save': args.n_save, 'n_print': n_print,
           'b_input': args.b_input, 'b_sample': args.b_sample, 'b_update': b_update, 'b_sync': args.b_sync}
    datasets = (tr_data, cv_data)
    train_hybrid(model, datasets, args.n_epoch, device, cfg, args.mix_loss,
              fp16=args.fp16, dist=dist)

def distill_s2s_model(model, teacher, args, device, n_device=1):
    dist, verbose = n_device > 1, device == 0
    tr_data = SpectroDataset(args.train_scp, args.train_target, downsample=args.downsample,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             spec_drop=args.spec_drop, spec_bar=args.spec_bar, spec_ratio=args.spec_ratio,
                             time_stretch=args.time_stretch, time_win=args.time_win,
                             threads=2, verbose=verbose)
    cv_data = SpectroDataset(args.valid_scp, args.valid_target, downsample=args.downsample,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             threads=2, verbose=verbose)
    if dist: tr_data.partition(device, n_device)
    n_print = args.n_print // n_device
    b_update = args.b_update // n_device

    cfg = {'model_path': args.model_path, 'lr': args.lr, 'grad_norm': args.grad_norm,
           'weight_decay': args.weight_decay, 'weight_noise': args.weight_noise,
           'label_smooth': args.label_smooth, 'teacher_force': args.teacher_force,
           'n_warmup': args.n_warmup, 'n_const': args.n_const, 'n_save': args.n_save, 'n_print': n_print,
           'b_input': args.b_input, 'b_sample': args.b_sample, 'b_update': b_update, 'b_sync': args.b_sync}
    datasets = (tr_data, cv_data)
    distill_s2s(model, teacher, args.alpha, datasets, args.n_epoch, device, cfg, fp16=args.fp16, dist=dist)

def train_encoder_model(model, args, device, n_device=1):
    dist, verbose = n_device > 1, device == 0
    tr_data = SpectroDataset(args.train_scp, args.train_target, downsample=args.downsample, sek=False,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             spec_drop=args.spec_drop, spec_bar=args.spec_bar, spec_ratio=args.spec_ratio,
                             threads=2, verbose=verbose)
    cv_data = SpectroDataset(args.valid_scp, args.valid_target, downsample=args.downsample, sek=False,
                             sort_src=True, mean_sub=args.mean_sub, fp16=args.fp16, preload=args.preload,
                             threads=2, verbose=verbose)
    if dist: tr_data.partition(device, n_device)
    n_print = args.n_print // n_device
    b_update = args.b_update // n_device

    cfg = {'model_path': args.model_path, 'lr': args.lr, 'grad_norm': args.grad_norm,
           'weight_decay': args.weight_decay, 'weight_noise': args.weight_noise,
           'n_warmup': args.n_warmup, 'n_const': args.n_const, 'n_save': args.n_save, 'n_print': n_print,
           'b_input': args.b_input, 'b_sample': args.b_sample, 'b_update': b_update, 'b_sync': args.b_sync}
    datasets = (tr_data, cv_data)
    train_encoder(model, datasets, args.n_epoch, device, cfg, fp16=args.fp16, dist=dist)
    
def train_text_encoder_model(model, args, device, n_device=1):
    dist, verbose = n_device > 1, device == 0
    tr_data = TextPairDataset(args.train_src, args.train_tgt, threads=2, verbose=verbose)
    cv_data = TextPairDataset(args.valid_src, args.valid_tgt, threads=2, verbose=verbose)
    if dist: tr_data.partition(device, n_device)
    n_print = args.n_print // n_device
    b_update = args.b_update // n_device

    cfg = {'model_path': args.model_path, 'lr': args.lr, 'grad_norm': args.grad_norm,
           'weight_decay': args.weight_decay, 'weight_noise': args.weight_noise,
           'n_warmup': args.n_warmup, 'n_const': args.n_const, 'n_save': args.n_save, 'n_print': n_print,
           'b_input': args.b_input, 'b_sample': args.b_sample, 'b_update': b_update, 'b_sync': args.b_sync}
    datasets = (tr_data, cv_data)
    train_encoder(model, datasets, args.n_epoch, device, cfg, fp16=args.fp16, dist=dist)
