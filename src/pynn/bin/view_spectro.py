#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import os
import random
import numpy as np
import argparse

import matplotlib.pyplot as plt

from pynn.io.audio_seq import SpectroDataset

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--downsample', help='concated frames', type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()

    reader = SpectroDataset(args.data_scp, mean_sub=args.mean_sub)
    while True:
        src, mask, utts = reader.read_batch_utt(1)
        if not utts: break

        utt, img = utts[0], src[0]
        if args.downsample > 1:
            import torch
            import torch.nn.functional as F
            size = (img.shape[0] // args.downsample, img.shape[1])
            print(size)
            img = F.interpolate(img.view(1, 1, img.size(0), img.size(1)), size=size)
            img = img.squeeze(0).squeeze(0)
        img = img.numpy()
        plt.imshow(img.T)
        plt.savefig('%s.png' % utt)
