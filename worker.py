# cython: language_level=3
from MCloud import MCloudWrap
from MCloud import mcloudpacketdenit, mcloudpacketdenit
from MCloudPacketRCV import MCloudPacketRCV

import os
import torch
import numpy as np
import time
import threading
import argparse

from pynn.util import audio
from decoder import init_asr_model, decode, token2word, clean_noise
from decoder import init_punct_model, token2punct

from segmenter import Segmenter

def segmenter_thread(mcloud):
    stream_id = "123456789".encode("utf-8")

    connected = 1
    while True:
        try:
            if connected == 1:
                # connect to mediator
                connected = mcloud.connect(serverHost.encode("utf-8"), serverPort)
                if connected == 1:
                    print("ERROR Could not connect to the Mediator. Reconnect in 10 seconds.")
                    time.sleep(10); continue
                else:
                    print("WORKER INFO Connection established ==> waiting for clients.")
        
            # wait for client
            if mcloud.wait_for_client(stream_id) == 1:
                print("WORKER ERROR while waiting for client")
                time.sleep(1); connected = 1; continue
            else:
                print("WORKER INFO received client request ==> waiting for packages")

            proceed = True
            while (proceed):
                packet = MCloudPacketRCV(mcloud)

                type = packet.packet_type()
                if packet.packet_type() == 3:
                    mcloud.process_data_async(packet, data_callback)
                elif packet.packet_type() == 7:  # MCloudFlush
                    """
                    a flush message has been received -> wait (block) until all pending packages
                    from the processing queue has been processed -> finalizeCallback will
                    be called-> flush message will be passed to subsequent components
                    """
                    mcloud.wait_for_finish(0, "processing")
                    mcloud.send_flush()
                    print("WORKER INFO received flush message ==> waiting for packages.")
                    mcloudpacketdenit(packet)
                    break
                elif packet.packet_type() == 4:  # MCloudDone
                    print("WOKRER INFO received DONE message ==> waiting for clients.")
                    mcloud.wait_for_finish(1, "processing")
                    mcloud.stop_processing("processing")
                    mcloudpacketdenit(packet)
                    proceed = False
                elif packet.packet_type() == 5:  # MCloudError
                    # In case of a error or reset message, the processing is stopped immediately by
                    # calling mcloudBreak followed by exiting the thread.
                    mcloud.wait_for_finish(1, "processing")
                    mcloud.stop_processing("processing")
                    mcloudpacketdenit(packet)
                    print("WORKER INFO received ERROR message >>> waiting for clients.")
                    proceed = False
                elif packet.packet_type() == 6:  # MCloudReset
                    mcloud.stop_processing("processing")
                    print("CLIENT INFO received RESET message >>> waiting for clients.")
                    mcloudpacketdenit(packet)
                    proceed = False
                else:
                    print("CLIENT ERROR unknown packet type {!s}".format(packet.packet_type()))
                    proceed = False
            print("WORKER WARN connection terminated ==> trying to reconnect.")
        except:
            print("Unexpected error. Reconnect in 10 seconds.")
            time.sleep(10)
    
def processing_finalize_callback():
    print("INFO in processing finalize callback")
    segmenter.reset()

def processing_error_callback():
    print("INFO In processing error callback")

def processing_break_callback():
    print("INFO in processing break callback")

def init_callback():
    print("INFO in processing init callback ")
    segmenter.reset()

def data_callback(i,sampleA):
    sample = np.asarray(sampleA, dtype=np.int16)
    segmenter.append_signal(sample.tobytes())
    return 0

def insert_new_words(model, args):
    while True:
        words = []
        with open(args.new_words,"r") as f:
            for line in f:
                line = line.strip()
                words.append(line)

        model.new_words(words)

        time.sleep(1)

def send_hypo(start, end, hypo, output):
    lh = 0 if output=='text' else len(hypo)
    mcloud_w.send_packet_result_async(start, end, hypo, lh)

class Segment(object):
    """Represents a hypothesis segment """
    def __init__(self, hypo, start, end, text=[], fixed=False, final=False):
        self.hypo = hypo[0]
        self.bmes = hypo[1]
        self.start = start
        self.end = end
        self.text = text
        self.fixed = fixed
        self.final = final
        self.uppercase = False
        self.hard_stop = False

    def __str__(self):
        return self.hypo

def send_segs(segs, output):
    if len(segs) == 0: return
    start, end = segs[0].start, segs[-1].end
    hypo = []
    for seg in segs: hypo.extend(seg.text)
    hypo = ['<unk>' if w.startswith('%') or w.startswith('+') or w.startswith('<') or \
            w.startswith('*') or w.endswith('*') else w for w in hypo]
    #print('Sending %d %d %s' % (start, end, ' '.join(hypo)))
    send_hypo(start, end, hypo, output)

def seg2text(segs):
    global count
    hypo = []
    for seg in segs: hypo.append(' '.join(seg.text))
    return '<%d> %s <%d>' % (segs[0].start, ' | '.join(hypo), segs[-1].end)

def punct_segs(punct, device, dic, space, segs):
    for j, seg in enumerate(segs):
        if seg.fixed: continue
        k, lctx = j, [] 
        while k > 0 and len(lctx) < 6:
            lctx = segs[k-1].hypo + lctx
            k -= 1
        k, rctx = j+1, []
        while k < len(segs) and len(rctx) < 6:
            rctx = rctx + segs[k].hypo
            k += 1
        text = token2punct(punct, device, seg, lctx, rctx, dic, space)
        seg.text = text
        if len(seg.text) > 0:
            if seg.uppercase: seg.text[0] = seg.text[0].capitalize()
            if seg.hard_stop: seg.text[-1] = seg.text[-1].capitalize()

        rctx = sum(len(seg.hypo) for seg in segs[j+1:])
        if rctx >= 10: seg.fixed = True

    pre_w = ''
    for seg in segs:
        for j, word in enumerate(seg.text):
            if pre_w.endswith('.') or pre_w.endswith('!') or pre_w.endswith('?'): # or pre_w.endswith(':'):
                seg.text[j] = word.capitalize()
            pre_w = word

    for seg, seg_nx in zip(segs[:-1], segs[1:]):
        if seg_nx.fixed: seg.final = True

    return segs


def update_and_send(punct, device, dic, space, segs, new_seg, wc):
    if len(segs) > 0:
        last_seg = segs[-1]
        if last_seg.start < new_seg.start:
            segs.append(new_seg)
        else:
            new_seg.uppercase = last_seg.uppercase
            segs[-1] = new_seg
    else:
        segs.append(new_seg)

    old_segs, new_segs = [], []   
    for j in range(len(segs)-1):
        seg, seg_nx = segs[j], segs[j+1]
        if seg.end < seg_nx.start - 2*1000:
            old_segs, new_segs = segs[:j+1], segs[j+1:]
            old_segs[-1].hard_stop = True
            old_segs[-1].fixed = False
            new_segs[0].uppercase = False
            new_segs[0].fixed = False
            break
    if len(old_segs) == 0:
        clean_time = segs[-1].start - 6*1000 # 6 seconds
        j = 0
        while segs[j].end < clean_time or segs[j].final: j+= 1
        old_segs, new_segs = segs[:j], segs[j:]

    if len(old_segs) > 0:
        if not old_segs[-1].fixed:
            old_segs = punct_segs(punct, device, dic, space, old_segs)
        print('Sending final: ' + seg2text(old_segs))

        for seg in old_segs: wc += len(seg.text)
        if wc > 100:
            br = False
            for j in range(len(old_segs)-1, -1, -1):
                seg = old_segs[j]
                for k in range(len(seg.text)-1, -1, -1):
                    if seg.text[k].endswith('.'):
                        seg.text[k] = seg.text[k] + '<br><br>'
                        br = True; wc = 0; break
                if br: break
        send_segs(old_segs, args.outputType)

    segs = punct_segs(punct, device, dic, space, new_segs)
    print('Sending partial: ' + seg2text(segs))
    send_segs(segs, args.outputType)

    punct.model.lock.release()

    return segs, wc

parser = argparse.ArgumentParser(description='pynn')
#model argument
parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--punct-dic', help='dictionary file', default=None)
parser.add_argument('--device', help='device', type=str, default='cuda')
parser.add_argument('--beam-size', help='beam size', type=int, default=8)
parser.add_argument('--attn-head', help='attention head', type=int, default=0)
parser.add_argument('--attn-padding', help='attention padding', type=float, default=0.05)
parser.add_argument('--stable-time', help='stable size', type=int, default=200)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--prune', help='pruning threshold', type=float, default=1.0)
parser.add_argument('--incl-block', help='incremental block size', type=int, default=50)
parser.add_argument('--max-len', help='max length', type=int, default=100)
parser.add_argument('--space', help='space token', type=str, default='▁')
parser.add_argument('--seg-based', help='output when audio segment is complete', action='store_true')
parser.add_argument('--new-words', help='path to text file with new words', default="words.txt")

#worker argument
parser.add_argument('-s','--server', type=str, default="i13srv53.ira.uka.de")
parser.add_argument('-p','--port' ,type=int, default=60019)
parser.add_argument('-n','--name' ,type=str, default="asr-EN")
parser.add_argument('-fi','--fingerprint', type=str, default="en-EU")
parser.add_argument('-fo','--outfingerprint',type=str, default="en-EU")
parser.add_argument('-i','--inputType' ,type=str, default="audio")
parser.add_argument('-o','--outputType', type=str, default="unseg-text")
args = parser.parse_args()
print(args)

serverHost = args.server
serverPort = args.port
worker_name = args.name
inputFingerPrint  = args.fingerprint
inputType         = args.inputType
outputFingerPrint = args.outfingerprint
outputType        = args.outputType
specifier           = ""

sample_rate = 16000
VAD_aggressive = 2
padding_duration_ms = 450
frame_duration_ms = 30
rate_begin = 0.65
rate_end = 0.55
segmenter = Segmenter(sample_rate, VAD_aggressive, padding_duration_ms, frame_duration_ms, rate_begin, rate_end)

print("#" * 40 + " >> TESTING MCLOUD WRAPPER API << " + "#" * 40)
mcloud_w = MCloudWrap("asr".encode("utf-8"), 1)
mcloud_w.add_service(worker_name.encode("utf-8"), "asr".encode("utf-8"), inputFingerPrint.encode("utf-8"), inputType.encode("utf-8"),outputFingerPrint.encode("utf-8"), outputType.encode("utf-8"), specifier.encode("utf-8"))
#set callback
mcloud_w.set_callback("init", init_callback)
mcloud_w.set_data_callback("worker")
mcloud_w.set_callback("finalize", processing_finalize_callback)
mcloud_w.set_callback("error", processing_error_callback)
mcloud_w.set_callback("break", processing_break_callback)

#clean tempfile

fbank_mat = audio.filter_bank(sample_rate, 256, 40)
print("Initialize the model...")
model, device, dic = init_asr_model(args)
punct = init_punct_model(args)
punct.model = model
torch.set_grad_enabled(False)
print("Done.")

#segmentor thread
record = threading.Thread(target=segmenter_thread, args=(mcloud_w,))
record.start()

new_words = threading.Thread(target=insert_new_words, args=(model,args))
new_words.start()

try:
    cur_segs, wc = [], 0
    while True:
        if not segmenter.ready:
            time.sleep(0.2)
            continue
        segment, pos = segmenter.active_seg()
        if segment is None:
            time.sleep(0.1)
            continue

        ss = segment.start_time()
        #if len(cur_segs) > 0 and cur_segs[-1].end > ss*10:
        if ss == 0 or pos == 0:
            cur_segs, wc = [], 0 

        ntime = 600 # 1000
        phypo, shypo = [1], [1]
        h_start, h_end = 0, 0
        while not segment.completed and not args.seg_based:
            sec = segment.size()
            #print('Segment size: %d miliseconds' % sec)
            if sec > ntime:
                ntime += 600
                adc = segment.get_all()
                hypo, sp, ep, frames, best_memory_entry = decode(model, device, args, adc, fbank_mat, h_start, shypo)
                if len(hypo) == 0: continue

                for j in range(len(shypo), min(len(phypo), len(hypo))):
                    if phypo[j] != hypo[j]: break
                    shypo = hypo[:j]
                if shypo[-1] == 2: shypo = shypo[:-1]
                #print(shypo)
                phypo = hypo

                end = 0 if len(hypo)<=2 else ep[len(hypo)-2]

                start = ss + h_start*10
                if h_start + end > h_end:
                    h_end = h_start + end
                    if h_end > h_start+300:
                        j = len(shypo)-1
                        while j > 2:
                            if (sp[j-1]+16) >= ep[j-2] and dic[shypo[j]-2].startswith(args.space): break
                            j -= 1
                        if j > 5:
                            h_end = h_start + sp[j-1] - 1
                            hypo = shypo[:j] + [2]
                            best_memory_entry = best_memory_entry[:j]
                            h_start, shypo, phypo = h_end+1, [1]+shypo[j:], [1]+phypo[j:]
                    end = ss + h_end*10
                else:
                    end = start + frames*10

                hypo = hypo[1:-1]
                best_memory_entry = best_memory_entry[:-1]
                if punct is not None:
                    #end = start + frames*10
                    new_seg = Segment(clean_noise(hypo, best_memory_entry, dic, args.space), start, end)
                    cur_segs, wc = update_and_send(punct, device, dic, args.space, cur_segs, new_seg, wc)
                else:
                    hypo = token2word(hypo, dic, args.space)
                    print('Sending partial: ' + ' '.join(hypo))
                    send_hypo(start, end, hypo, args.outputType)

        if segment.completed:
            adc = segment.get_all()
            hypo, sp, ep, frames, best_memory_entry = decode(model, device, args, adc, fbank_mat, h_start, shypo)
            if len(hypo) == 0: continue
            start, end = ss + h_start*10, ss + (h_start+frames)*10

            hypo = hypo[1:-1]
            best_memory_entry = best_memory_entry[:-1]
            if punct is not None:
                new_seg = Segment(clean_noise(hypo, best_memory_entry, dic, args.space), start, end)
                cur_segs, wc = update_and_send(punct, device, dic, args.space, cur_segs, new_seg, wc)
            else:
                hypo = token2word(hypo, dic, args.space)
                print('Sending final: ' + ' '.join(hypo))
                send_hypo(start, end, hypo, args.outputType)
            
            segment.finish()
            print('Finished segment.')
            
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Terminating running threads..")

record.join()
new_words.join()
