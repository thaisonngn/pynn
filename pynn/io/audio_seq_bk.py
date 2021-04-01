# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
import struct
import os
import numpy as np

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from . import smart_open
 
class SpectroDataset(Dataset):
    def __init__(self, scp_path, label_path=None, paired_label=False,
                 verbose=True, sek=True, sort_src=False, pack_src=False,
                 downsample=1, preload=False, threads=4, fp16=False, 
                 spec_drop=False, spec_bar=2, spec_ratio=0.4,
                 time_stretch=False, time_win=10000, mean_sub=False):
        self.scp_path = scp_path     # path to the .scp file
        self.label_path = label_path # path to the label file
        self.paired_label = paired_label

        self.downsample = downsample
        self.sort_src = sort_src
        self.pack_src = pack_src
        self.sek = sek

        self.mean_sub = mean_sub
        self.spec_drop = spec_drop
        self.spec_bar = spec_bar
        self.spec_ratio = spec_ratio
        self.time_stretch = time_stretch
        self.time_win = time_win
        self.fp16 = fp16

        self.threads = threads
        self.preload = preload
        self.utt_lbl = None
        self.ark_cache = None
        self.ark_files = {}

        self.scp_file = None
        self.lbl_dic = None

        self.verbose = verbose
        self.rank = 0
        self.parts = 1
        self.epoch = -1

    def partition(self, rank, parts):
        self.rank = rank
        self.parts = parts
   
    def set_epoch(self, epoch):
        self.epoch = epoch

    def print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def initialize(self, b_input=20000, b_sample=64):
        if self.utt_lbl is not None:
            return

        path = os.path.dirname(self.scp_path)
        scp_dir = path + '/' if path != '' else ''

        utts = {}
        for line in smart_open(self.scp_path, 'r'):
            if line.startswith('#'): continue
            tokens = line.replace('\n','').split(' ')
            utt_id, path_pos = tokens[0:2]
            utt_len = -1 if len(tokens)<=2 else int(tokens[2])
            path, pos = path_pos.split(':')
            path = path if path.startswith('/') else scp_dir + path
            utts[utt_id] = (utt_id, path, pos, utt_len)
 
        labels = {}
        for line in smart_open(self.label_path, 'r'):
            tokens = line.split()
            utt_id = tokens[0]
            if utt_id == '' or utt_id not in utts: continue

            if self.paired_label:
                sp = tokens.index('|')
                lb1 = [int(token) for token in tokens[:sp]]
                lb1 = [1] + [el+2 for el in lb1] + [2] if self.sek else lb1
                lb2 = [int(token) for token in tokens[sp+1:]]
                lb2 = [1] + [el+2 for el in lb2] + [2] if self.sek else lb2
                lbl = (lb1, lb2)
            else:
                lbl = [int(token) for token in tokens[1:]]
                lbl = [1] + [el+2 for el in lbl] + [2] if self.sek else lbl
            labels[utt_id] = lbl

        utt_lbl = []
        for utt_id, utt_info in utts.items():
            if utt_id not in labels: continue
            utt_lbl.append([*utt_info, labels[utt_id]])
        self.utt_lbl = utt_lbl
        self.print('%d label sequences loaded.' % len(self.utt_lbl))
        self.print('Creating batches.. ', end='')
        self.batches = self.create_batch(b_input, b_sample)
        self.print('Done.')
        
        if self.preload:
            self.print('Loading ark files.. ', end='')
            self.preload_feats()
            self.print('Done.')

    def create_loader(self):
        batches = self.batches.copy()
        if self.epoch > -1:
            random.seed(self.epoch)
            random.shuffle(batches)
        if self.parts > 1:
            l = (len(batches) // self.parts) * self.parts
            batches = [batches[j] for j in range(self.rank, l, self.parts)]
        
        loader = DataLoader(self, batch_sampler=batches, collate_fn=self.collate_fn,
                            num_workers=self.threads, pin_memory=False)
        return loader

    def create_batch(self, b_input, b_sample):
        utts = self.utt_lbl
        for utt in utts:
            path, pos, utt_len = utt[1:4]
            if utt_len < 0: utt[3] = self._read_length(path, pos, cache=True)
        self._close_ark_files()

        lst = [(j, utt[3]) for j, utt in enumerate(utts)]
        lst = sorted(lst, key=lambda e : e[1])

        s, j, step = 0, 4, 4
        batches = []
        while j <= len(lst):
            bs = j - s
            if lst[j-1][1]*bs < b_input and bs < b_sample:
                j += step
                continue
            if bs > 8: j = s + (bs // 8) * 8
            batches.append([idx for idx, _ in lst[s:j]])
            s = j
            j += step
        if s < len(lst): batches.append([idx for idx, _ in lst[s:]])
        return batches

    def preload_feats(self):
        mats = {}
        for utt in self.utt_lbl:
            utt_id, path, pos = utt[0:3]
            mat = self._read_mat(path, pos, cache=True)
            mats[utt_id] = mat
        self.ark_cache = mats
        self._close_ark_files()
 
    def _read_string(self, ark_file):
        s = ''
        while True:
            c = ark_file.read(1).decode('utf-8')
            if c == ' ' or c == '': return s
            s += c

    def _read_integer(self, ark_file):
        n = ord(ark_file.read(1))
        return struct.unpack('>i', ark_file.read(n)[::-1])[0]

    def _read_length(self, path, pos, cache=False):
        if cache and path in self.ark_files:
            ark_file = self.ark_files[path]
        else:
            ark_file = smart_open(path, 'rb')
            if cache: self.ark_files[path] = ark_file

        ark_file.seek(int(pos))
        header = ark_file.read(2).decode('utf-8')
        if header != "\0B":
            raise Exception("Input .ark file is not binary")
        format = self._read_string(ark_file)
        utt_len = self._read_integer(ark_file) if format in ('FM', 'HM') else -1

        if not cache: ark_file.close()
        return utt_len

    def _close_ark_files(self):
        for fi in self.ark_files.values(): fi.close()
        self.ark_files = {}

    def _read_mat(self, path, pos, cache=False):
        if cache and path in self.ark_files:
            ark_file = self.ark_files[path]
        else:
            ark_file = smart_open(path, 'rb')
            if cache: self.ark_files[path] = ark_file

        ark_file = smart_open(path, 'rb')

        ark_file.seek(int(pos))
        header = ark_file.read(2).decode('utf-8')
        if header != "\0B": return None

        format = self._read_string(ark_file)
        if format == "FM" or format == "HM":
            rows = self._read_integer(ark_file)
            cols = self._read_integer(ark_file)
            fm, dt, sz = ("<%df", np.float32, 4) if format == "FM" else ("<%de", np.float16, 2)
            utt_mat = struct.unpack(fm % (rows * cols), ark_file.read(rows*cols*sz))
            utt_mat = np.array(utt_mat, dtype=dt)
            if self.fp16 and dt == np.float32:
                utt_mat = utt_mat.astype(np.float16)
            utt_mat = np.reshape(utt_mat, (rows, cols))
        else:
            utt_mat = None

        if not cache: ark_file.close()
        return utt_mat

    def __len__(self):
        return len(self.utt_lbl)

    def __getitem__(self, index):
        utt_id, path, pos, _, lbl  = self.utt_lbl[index]
        utt_mat = self.read_mat_cache(utt_id, path, pos)
        return (utt_mat, lbl)

    def read_mat_cache(self, utt_id, path, pos):
        cache = self.ark_cache
        if cache is not None and utt_id in cache:
            return cache[utt_id]
        return self._read_mat(path, pos)

    def read_utt(self):
        if self.scp_file is None:
            self.scp_file = smart_open(self.scp_path, 'r')
            path = os.path.dirname(self.scp_path)
            self.scp_dir = path + '/' if path != '' else ''

        line = self.scp_file.readline()
        if not line: return None, None
        utt_id, path_pos = line.replace('\n','').split(' ')[0:2]
        path, pos = path_pos.split(':')
        path = path if path.startswith('/') else self.scp_dir + path
        utt_mat = self._read_mat(path, pos)
        return utt_id, utt_mat

    def read_batch_utt(self, batch_size=10):
        mats, ids = [], []
        for i in range(batch_size):
            utt_id, utt_mat = self.read_utt()
            if utt_id is None or utt_id == '': break
            mats.append(utt_mat)
            ids.append(utt_id)
        if len(mats) == 0: return (None, None, None)

        lst = sorted(zip(mats, ids), key=lambda e : -e[0].shape[0])
        src, ids = zip(*lst)
        src = self.augment_src(src)
        src = self.collate_src(src)
        return (*src, ids)

    def read_label(self, utt_id):
        if self.lbl_dic is None:
            lbl_dic = {}
            for line in smart_open(self.label_path, 'r'):
                tokens = line.split()
                utt_id = tokens[0]
                if utt_id == '': continue
                lbl_dic[utt_id] = [int(token) for token in tokens[1:]]
            self.lbl_dic = lbl_dic

        if utt_id not in self.label_dic:
            return None

        utt_lbl = self.label_dic[utt_id]
        if self.sek:
            utt_lbl = [1] + [el+2 for el in utt_lbl] + [2]
        return utt_lbl

    def timefreq_drop_inst(self, inst, num=2, time_drop=0.4, freq_drop=0.4):
        time_num, freq_num = inst.shape
        freq_num = freq_num

        n = random.randint(0, int(freq_drop*freq_num))
        f0 = random.randint(0, freq_num-n)
        inst[:, f0:f0+n] = 0

        max_time = int(time_drop * time_num)
        num = random.randint(1, num)
        time_len = max_time // num
        for i in range(num):
            n = min(max_time, random.randint(0, time_len))
            t0 = random.randint(0, time_num-n)
            inst[t0:t0+n, :] = 0    

        return inst

    def time_stretch_inst(self, inst, low=0.85, high=1.2, win=10000):
        time_len = inst.shape[0]
        ids = None
        for i in range((time_len // win) + 1):
            s = random.uniform(low, high)
            e = min(time_len, win*(i+1))          
            r = np.arange(win*i, e-1, s, dtype=np.float32)
            r = np.round(r).astype(np.int32)
            ids = r if ids is None else np.concatenate((ids, r))
        return inst[ids]

    def mean_sub_inst(self, inst):
        return inst - inst.mean(axis=0, keepdims=True)

    def down_sample_inst(self, feats, cf=4):
        feats = feats[:(feats.shape[0]//cf)*cf,:]
        return feats.reshape(feats.shape[0]//cf, feats.shape[1]*cf)
     
    def augment_src(self, src):
        insts = []
        bar, ratio = self.spec_bar, self.spec_ratio
        for inst in src:
            inst = self.mean_sub_inst(inst) if self.mean_sub else inst
            inst = self.time_stretch_inst(inst, win=self.time_win) if self.time_stretch else inst
            inst = self.timefreq_drop_inst(inst, num=bar, time_drop=ratio) if self.spec_drop else inst            
            inst = self.down_sample_inst(inst, self.downsample) if self.downsample > 1 else inst
            insts.append(inst)
        return insts

    def collate_src(self, insts):
        max_len = max(inst.shape[0] for inst in insts)
        inputs = np.zeros((len(insts), max_len, insts[0].shape[1]))
        masks = torch.zeros((len(insts), max_len), dtype=torch.uint8)
        
        for idx, inst in enumerate(insts):
            inputs[idx, :inst.shape[0], :] = inst
            masks[idx, :inst.shape[0]] = 1
        inputs = torch.HalfTensor(inputs) if self.fp16 else torch.FloatTensor(inputs)

        return inputs, masks

    def collate_src_pack(self, insts):
        max_len = max(inst.shape[0] for inst in insts)
        masks = torch.zeros((len(insts), max_len), dtype=torch.uint8)
        inputs = []
        
        for idx, inst in enumerate(insts):
            inputs.append(torch.HalfTensor(inst) if self.fp16 else torch.FloatTensor(inst))
            masks[idx, 0:inst.shape[0]] = 1
        inputs = pack_sequence(inputs)
                    
        return inputs, masks

    def collate_tgt(self, tgt):
        if self.paired_label:
            lb1, lb2 = zip(*tgt)
            max_len = max(len(inst) for inst in lb1)
            lb1 = np.array([inst + [0] * (max_len - len(inst)) for inst in lb1])
            max_len = max(len(inst) for inst in lb2)
            lb2 = np.array([inst + [0] * (max_len - len(inst)) for inst in lb2])
            labels = (torch.LongTensor(lb1), torch.LongTensor(lb2))
        else:
            max_len = max(len(inst) for inst in tgt)
            labels = np.array([inst + [0] * (max_len - len(inst)) for inst in tgt])
            labels = (torch.LongTensor(labels),)

        return (*labels,)

    def collate_fn(self, batch):
        src, tgt = zip(*batch)
        src = self.augment_src(src)

        if self.sort_src or self.pack_src:
            lst = sorted(zip(src, tgt), key=lambda e : -e[0].shape[0])
            src, tgt = zip(*lst)

        src = self.collate_src(src) if not self.pack_src else self.collate_src_pack(src)
        tgt = self.collate_tgt(tgt)

        return (*src, *tgt)
