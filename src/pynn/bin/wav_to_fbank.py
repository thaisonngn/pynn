#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse
import multiprocessing
from scipy.io import wavfile

from pynn.util import audio
from pynn.io import kaldi_io

def write_ark_thread(segs, out_ark, out_scp, args):
    fbank_mat = audio.filter_bank(args.sample_rate, args.nfft, args.fbank)
    cache_wav = ''
    
    for seg in segs:
        tokens = seg.split()
        
        if args.seg_name:
            wav, start, end = tokens[:3]
            seg_name = '%s-%06.f-%06.f' % (wav, float(start)*100, float(end)*100)
        else:
            if len(tokens) == 2: tokens.extend(['0.0', '0.0'])
            seg_name, wav, start, end = tokens[:4]
        start, end = float(start), float(end)
        
        if args.wav_path is not None:
            wav = wav if wav.endswith('.wav') else wav + '.wav'
            wav = args.wav_path + '/' + wav
        if cache_wav != wav:
            sample_rate, signal = wavfile.read(wav)
            if sample_rate != args.sample_rate:
                print('Wav %d is not in desired sample rate' % wav)
                continue
            cache_wav = wav

        start = int(start * sample_rate)
        end = -1 if end <= 0. else int(end * sample_rate)
        feats = audio.extract_fbank(signal[start:end], fbank_mat, sample_rate=sample_rate)
        
        if args.mean_norm:
            feats = feats - feats.mean(axis=0, keepdims=True)
        
        dic = {seg_name: feats}
        kaldi_io.write_ark(out_ark, dic, out_scp, append=True)
    

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--seg-desc', help='path to segment description file', required=True)
parser.add_argument('--seg-name', help='generate segment name with timestamps', action='store_true')
parser.add_argument('--wav-path', help='path to wav files', type=str, default=None)
parser.add_argument('--sample-rate', help='path to wav files', type=int, default=16000)
parser.add_argument('--fbank', help='number of filter banks', type=int, default=40)
parser.add_argument('--nfft', help='number of FFT points', type=int, default=256)
parser.add_argument('--mean-norm', help='to perform mean substraction', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='data')
parser.add_argument('--jobs', help='number of parallel jobs', type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    
    segs = [line.rstrip('\n') for line in open(args.seg_desc, 'r')]
    size = len(segs) // args.jobs
    jobs = []
    j = 0
    for i in range(args.jobs):
        l = len(segs) if i == (args.jobs-1) else j+size
        sub_segs = segs[j:l]
        j += size
        out_ark = '%s.%d.ark' % (args.output, i)
        out_scp = '%s.%d.scp' % (args.output, i)
         
        process = multiprocessing.Process(
                target=write_ark_thread, args=(sub_segs, out_ark, out_scp, args))
        process.start()
        jobs.append(process)
    
    for job in jobs: job.join()
    
