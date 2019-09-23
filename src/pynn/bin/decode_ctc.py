
import sys
import threading
import argparse

import torch
import torch.nn.functional as F

from pynn.io.kaldi import KaldiStreamLoader
from pynn.net.rnn_ctc import DeepLSTMv2
from pynn.util.decoder import Decoder

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--data-scp', help='path to data scp file', required=True)
parser.add_argument('--model-file', help='model file', required=True)
parser.add_argument('--model-spec', help='model specification', required=True)
parser.add_argument('--bidirect', help='uni or bi directional', default='True')
parser.add_argument('--dict', help='dictionary file', required=True)

def parse_utt(utt):
    timestamp = utt[-13:]
    stime = timestamp[:6]
    stime = stime[:-2].lstrip('0') + '.' + stime[-2:]
    etime = timestamp[7:]
    etime = etime[:-2].lstrip('0') + '.' + etime[-2:]
    channel = utt[-15:-14]
    conv = utt[:-16] + channel
    return (conv, stime, etime)
    
if __name__ == '__main__':
    args = parser.parse_args()

    fin = open(args.dict, 'r')
    dic = {}
    for line in fin:
        tokens = line.split()
        dic[int(tokens[1])] = tokens[0]
    
    model_spec = map(int, args.model_spec.split(':'))
    model = DeepLSTMv2(input_size=model_spec[0], hidden_size=model_spec[1],
                layers=model_spec[2], num_classes=model_spec[3], bidirectional=args.bidirect)    
    model.load_state_dict(torch.load(args.model_file))
    model.train(False)
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    
    data_loader = KaldiStreamLoader(args.data_scp)
    data_loader.initialize()

    batch_size = 20
    ctm = open("hypos/H_1_LV.ctm", 'w')
    while True:
        inputs, input_sizes, utts = data_loader.read_batch_utt(batch_size)
        if len(inputs) == 0: break
        if use_gpu: inputs = inputs.cuda()
        
        outputs = model.extract(inputs, input_sizes)
        outputs = F.softmax(outputs, -1)
        
        print outputs.size()
        hypos = Decoder.decode_prob(outputs)
        for i in range(len(hypos)):
            hypo = hypos[i]
            if len(hypo) == 0: continue
            conv, stime, etime = parse_utt(utts[i])
            stime = float(stime)
            dur = (float(etime) - stime) / input_sizes[i]
            for token, frame, prob in hypo:
                if token == 0: continue
                #if float(prob) < 0.5: continue
                word = '<unk>'
                if token > 1: word = dic[token]
                #if float(prob) < 0.3: word = '<unk>'
                wtime = stime + frame*dur - 0.01
                ctm.write('%s 1 %.2f %.2f %s \t %.2f\n' % (conv, wtime, 0.02, word, prob))
    ctm.close()
