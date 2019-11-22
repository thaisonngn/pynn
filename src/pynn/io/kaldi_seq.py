# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
import struct
import os
import numpy as np
import torch

from torch.nn.utils.rnn import pack_sequence

from . import smart_open
 
class ScpStreamReader(object):

    def __init__(self, scp_path, label_path=None, time_idx_path=None, sek=True, downsample=1,
                       sort_src=False, pack_src=False, max_len=10000, max_utt=4096, 
                       mean_sub=False, zero_pad=0,
                       spec_drop=False, spec_bar=2, time_stretch=False, time_win=10000,
                       sub_seq=0.25, ss_static=False, shuffle=False, fp16=False):
        self.scp_path = scp_path     # path to the .scp file
        self.label_path = label_path # path to the label file
        self.time_idx_path = time_idx_path

        self.downsample = downsample
        self.shuffle = shuffle
        self.sort_src = sort_src
        self.pack_src = pack_src
        self.max_len = max_len
        self.sek = sek

        self.mean_sub = mean_sub
        self.zero_pad = zero_pad
        self.spec_drop = spec_drop
        self.spec_bar = spec_bar
        self.time_stretch = time_stretch
        self.time_win = time_win
        self.sub_seq = sub_seq
        self.ss_static = ss_static
        self.fp16 = fp16

        self.scp_dir = ''
        self.scp_file = None
        self.scp_pos = 0
        self.max_utt = max_utt
        
        # store features for each data partition
        self.feat = []
        self.label = []
        # store all label in a dictionary
        self.label_dic = None
        self.time_idx = None
        self.end_reading = False

    # read the feature matrix of the next utterance
    def read_all_label(self):
        self.label_dic = {}
        label_file = smart_open(self.label_path, 'r')
        for line in label_file:
            tokens = line.split()
            utt_id = tokens[0]
            if utt_id == '' or utt_id is None: continue
            self.label_dic[utt_id] = [int(token) for token in tokens[1:]]

    def read_time_index(self):
        self.time_idx = {}
        idx_file = smart_open(self.time_idx_path, 'r')
        for line in idx_file:
            tokens = line.split()
            utt_id = tokens[0]
            token = [int(token) for token in tokens[1:]]
            l = len(token) // 2
            self.time_idx[utt_id] = (token[0:l], token[l:])
        self.time_cache = ({}, {}, {})

    def _read_string(self, ark_file):
        s = ''
        while True:
            c = ark_file.read(1).decode('utf-8')
            if c == ' ' or c == '': return s
            s += c

    def _read_integer(self, ark_file):
        n = ord(ark_file.read(1))
        return struct.unpack('>i', ark_file.read(n)[::-1])[0]

    def initialize(self):
        if self.scp_file is None:
            self.scp_file = [line.rstrip('\n') for line in smart_open(self.scp_path, 'r')]
            path = os.path.dirname(self.scp_path)
            self.scp_dir = path + '/' if path != '' else None
        self.scp_pos = 0
        if self.shuffle: random.shuffle(self.scp_file)
        
        if self.label_path is not None and self.label_dic is None:
            self.read_all_label()
            print("Loaded labels of %d utterances" % len(self.label_dic))

        if self.time_idx_path is not None and self.time_idx is None:
            self.read_time_index()

        self.utt_index = 0
        if self.max_utt < 0 and len(self.feat) > 0:
            self.utt_count = len(self.feat)
        else:
            self.utt_count = 0
            self.end_reading = False

    # read the feature matrix of the next utterance
    def read_next_utt(self):
        if self.scp_pos >= len(self.scp_file):
            return '', None
        line = self.scp_file[self.scp_pos]
        utt_id, path_pos = line.replace('\n','').split(' ')
        path, pos = path_pos.split(':')
        if not path.startswith('/') and self.scp_dir is not None:
            path = self.scp_dir + path
        self.scp_pos += 1

        ark_file = smart_open(path, 'rb')
        ark_file.seek(int(pos))

        header = ark_file.read(2).decode('utf-8')
        if header != "\0B":
            print("Input .ark file is not binary"); exit(1)
        format = self._read_string(ark_file)
        
        if format == "FM":
            rows = self._read_integer(ark_file)
            cols = self._read_integer(ark_file) 
            #print rows, cols
            utt_mat = struct.unpack("<%df" % (rows * cols), ark_file.read(rows*cols*4))
            utt_mat = np.array(utt_mat, dtype="float32")
            if self.fp16:
                utt_mat = utt_mat.astype("float16")
            if self.zero_pad > 0:
                rows += self.zero_pad
                utt_mat.resize(rows*cols)
            utt_mat = np.reshape(utt_mat, (rows, cols))
        else:
            print("Unsupported .ark file with %s format" % format); exit(1)
        ark_file.close()

        return utt_id, utt_mat

    def read_batch_utt(self, batch_size=32):
        feats = []
        ids = []
        
        i = 0
        while i < batch_size:
            utt_id, utt_mat = self.read_next_utt()
            if utt_id is None or utt_id == '': break

            feats.append(utt_mat)
            ids.append(utt_id)
            i += 1
        if len(feats) == 0: return ([], [], [])

        lst = sorted(zip(feats, ids), key=lambda e : -e[0].shape[0])
        src, ids = zip(*lst)

        src = self.augment_src(src)
        src = self.collate_src(src)
        return (*src, ids)

    def read_utt_label(self, utt_id, utt_mat):
        if not utt_id in self.label_dic:
            #print('Labels not found for %s' % utt_id)
            return utt_mat, None

        if len(utt_mat) >= self.max_len:
            return utt_mat, None

        utt_lbl = self.label_dic[utt_id]

        if self.time_idx is not None and self.sub_seq > 0.:
            utt_mat, utt_lbl = self.sub_sequence_inst(utt_mat, utt_lbl, utt_id, self.sub_seq)
        
        if self.sek and utt_lbl is not None:
            utt_lbl = [1] + [el+2 for el in utt_lbl] + [2]
        return utt_mat, utt_lbl

    def next_partition(self):
        if self.end_reading:
            return 0

        self.feat = []
        self.label = []
        while self.max_utt < 0 or len(self.feat) < self.max_utt:
            utt_id, utt_mat = self.read_next_utt()
            if utt_id == '':    # No more utterances available
                self.end_reading = True
                break
            utt_mat, utt_lbl = self.read_utt_label(utt_id, utt_mat)
            if utt_lbl is None: continue            

            self.feat.append(utt_mat)
            self.label.append(utt_lbl)
        return len(self.feat)

    def available(self):
        if self.utt_index >= self.utt_count:
            self.utt_count = self.next_partition()
            self.utt_index = 0

        return self.utt_index < self.utt_count

    def sub_sequence_inst_(self, inst, tgt, utt_id, ratio):
        sx, fx = self.time_idx[utt_id]
        l = len(sx)        
        if l < 4 or random.random() > ratio:
            return inst, tgt
        sid = random.randint(1, l//4)
        eid = random.randint(l*3//4, l-1)
        tgt = tgt[sx[sid]:sx[eid]-1]
        inst = inst[fx[sid]:fx[eid]-1, :]
 
        return inst, tgt

    def sub_sequence_inst(self, inst, tgt, utt_id, ratio):
        if self.ss_static:
            return self.sub_sequence_inst_static(inst, tgt, utt_id, ratio)
            
        sx, fx = self.time_idx[utt_id] 
        l = len(sx)
        
        if l < 4:
            if random.random() < ratio: tgt = None
            return inst, tgt

        if random.random() > ratio:
            return inst, tgt

        mode = random.randint(0, 2)
        if mode == 0:
            idx = random.randint(l//2, l-1)
            tgt = tgt[0: sx[idx]-1]
            inst = inst[0:fx[idx]-1, :]
        elif mode == 1:
            idx = random.randint(1, l//2)
            tgt = tgt[sx[idx]:]
            inst = inst[fx[idx]:, :]
        elif mode == 2:
            sid = random.randint(1, l//4)
            eid = random.randint(l*3//4, l-1) 
            tgt = tgt[sx[sid]:sx[eid]-1]
            inst = inst[fx[sid]:fx[eid]-1, :]

        return inst, tgt

    def sub_sequence_inst_static(self, inst, tgt, utt_id, ratio):
        sx, fx = self.time_idx[utt_id] 
        l = len(sx)

        if l < 4:
            if random.random() < ratio: tgt = None
            return inst, tgt

        if random.random() > ratio:
            return inst, tgt

        mode = random.randint(0, 2)
        if mode == 0:
            if not utt_id in self.time_cache[mode]:
                idx = random.randint(l//2, l-1)
                self.time_cache[mode][utt_id] = idx
            else:
                idx = self.time_cache[mode][utt_id]
            tgt = tgt[0: sx[idx]-1]
            inst = inst[0:fx[idx]-1, :]
        elif mode == 1:
            if not utt_id in self.time_cache[mode]:
                idx = random.randint(1, l//2)
                self.time_cache[mode][utt_id] = idx
            else:
                idx = self.time_cache[mode][utt_id]
            tgt = tgt[sx[idx]:]
            inst = inst[fx[idx]:, :]
        elif mode == 2:
            if not utt_id in self.time_cache[mode]:
                sid = random.randint(1, l//4)
                eid = random.randint(l*3//4, l-1)        
                self.time_cache[mode][utt_id] = (sid, eid)
            else:
                sid, eid = self.time_cache[mode][utt_id]

            tgt = tgt[sx[sid]:sx[eid]-1]
            inst = inst[fx[sid]:fx[eid]-1, :]

        return inst, tgt

    def timefreq_drop_inst(self, inst, num=2, time_drop=0.2, freq_drop=0.15):
        time_num, freq_num = inst.shape
        freq_num = freq_num
        time_len = 72

        max_time = int(time_drop*time_num)
        for i in range(num):
            n = min(max_time, random.randint(0, time_len))
            t0 = random.randint(0, time_num-n)
            inst[t0:t0+n, :] = 0
            max_time -= n

            n = random.randint(0, int(freq_drop*freq_num))
            f0 = random.randint(0, freq_num-n)
            inst[:, f0:f0+n] = 0
        return inst

    def time_stretch_inst(self, inst, low=0.8, high=1.25, win=10000):
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

    def down_sample_inst(self, feature, cf=4):
        feature = feature[:(feature.shape[0]//cf)*cf,:]
        return feature.reshape(feature.shape[0]//cf, feature.shape[1]*cf)
     
    def augment_src(self, src):
        insts = []
        for inst in src:
            inst = self.time_stretch_inst(inst, win=self.time_win) if self.time_stretch else inst
            inst = self.mean_sub_inst(inst) if self.mean_sub else inst
            inst = self.timefreq_drop_inst(inst, num=self.spec_bar) if self.spec_drop else inst            
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
        max_len = max(len(inst) for inst in tgt)
        labels = np.array([inst + [0] * (max_len - len(inst)) for inst in tgt])
        labels = torch.LongTensor(labels)
 
        return labels, 
    
    def next_batch(self, batch_size=16):
        src = self.feat[self.utt_index:self.utt_index+batch_size]
        tgt = self.label[self.utt_index:self.utt_index+batch_size]

        src = self.augment_src(src)
        if self.sort_src or self.pack_src:
            lst = sorted(zip(src, tgt), key=lambda e : -e[0].shape[0])
            src, tgt = zip(*lst)

        self.utt_index += len(src)

        src = self.collate_src(src) if not self.pack_src else self.collate_src_pack(src)
        tgt = self.collate_tgt(tgt)
        return (*src, *tgt)

    def next(self, batch_input=3000):
        l = len(self.feat)
        j = i = self.utt_index
        tgs = max_l = 0
        while j < l:
            max_l = max(max_l, self.feat[j].shape[0])
            if j > i and max_l*(j-i+1) > batch_input: break
            tgs += len(self.label[j]); j += 1
        j = ((j-i)//4 * 4 + i) if j > (i+4) else j
        last = (j==l)

        src, tgt = self.feat[self.utt_index:j], self.label[self.utt_index:j]
        src = self.augment_src(src)
        if self.sort_src or self.pack_src:
            lst = sorted(zip(src, tgt), key=lambda e : -e[0].shape[0])
            src, tgt = zip(*lst)

        seqs = len(src)
        self.utt_index += seqs

        src = self.collate_src(src) if not self.pack_src else self.collate_src_pack(src)
        tgt = self.collate_tgt(tgt)
        
        return (*src, *tgt, seqs, tgs, last)
        

class ScpBatchReader(ScpStreamReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shuffle = False

    def next_partition(self):
        if self.end_reading:
            return 0

        self.feat, self.label = [], []
        feats, labels = [], []

        num = 0
        while num < self.max_utt:
            if self.scp_pos >= len(self.scp_file):
                self.end_reading = True; break

            if self.scp_file[self.scp_pos].startswith('#'):
                if len(feats) > 0:
                    self.feat.append(feats)
                    self.label.append(labels)
                    num += len(feats)
                    feats, labels = [], []
                self.scp_pos += 1
                continue

            utt_id, utt_mat = self.read_next_utt()
            if utt_id == '': 
                self.end_reading = True; break

            utt_mat, utt_lbl = self.read_utt_label(utt_id, utt_mat)
            if utt_lbl is None: continue

            feats.append(utt_mat)
            labels.append(utt_lbl)
            
        return len(self.feat)
        
    def next(self, batch_input=1):
        src, tgt = self.feat[self.utt_index], self.label[self.utt_index]

        tgs = 0
        for tg in tgt: tgs += len(tg)
        
        src = self.augment_src(src)
        if self.sort_src or self.pack_src:
            lst = sorted(zip(src, tgt), key=lambda e : -e[0].shape[0])
            src, tgt = zip(*lst)
        
        seqs = len(src)
        self.utt_index += 1
        last = (self.utt_index == len(self.feat))

        src = self.collate_src(src) if not self.pack_src else self.collate_src_pack(src)
        tgt = self.collate_tgt(tgt)
        
        return (*src, *tgt, seqs, tgs, last)
