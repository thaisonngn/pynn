
import os
import torch
import numpy as np
import time
import threading
import argparse
import urllib.request

from pynn.util import audio
from decoder import init_asr_model, decode, token2word, clean_noise
from decoder import init_punct_model, token2punct

from segmenter import Segmenter

from typing             import Any, Dict, Optional, cast, List
from abc                import ABCMeta, abstractmethod
import pyaudio

import warnings
warnings.filterwarnings("ignore")

class BaseAdapter():
    def __init__(self, format: Any, chunksize: int = 1024) -> None:
        self.rate = 16000
        self.chunksize = chunksize
        self.format = format
        self.channel_count = 1
        self.chosen_channel: Optional[int] = None

    def available(self) -> bool:
        """
        Should return if the backend is available and print an error message
        if not. Called before setting input
        """
        return True

    @abstractmethod
    def get_stream(self, **kwargs) -> Any:
        """
        Should return the stream for reading
        """
        pass

    @abstractmethod
    def read(self, chunksize: Optional[int] = None) -> bytes:
        """
        Should return a chunk of bytes from the audio device.
        If chunksize is None, use internal value or any size.
        Might block depending on device.
        chunksize might be ignored
        """
        pass

    def chunk_modify(self, chunk: bytes) -> bytes:
        """
        Allows modifying of the chunk
        """
        return chunk

    @abstractmethod
    def cleanup(self) -> None:
        """
        Should be called after the session was closed
        """
        pass

    @abstractmethod
    def set_input(self, input: Any) -> None:
        """
        Should be called to set an input, which can be for example an id, string etc
        """
        pass

class PortaudioStream(BaseAdapter):
    def __init__(self, **kwargs) -> None:
        self.input_id: Optional[int]             = None
        self._stream:  Optional[pyaudio.Stream]  = None
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        super().__init__(format=pyaudio.paInt16, chunksize=kwargs["chunksize"])

    def get_stream(self, **kwargs) -> pyaudio.Stream:
        if self.input_id is None:
            raise BugException()
        if self._stream is None:
            p = self.pyaudio
            self._stream = p.open(
                format              = self.format,
                input_device_index  = self.input_id,
                channels            = self.channel_count,
                rate                = self.rate,
                input               = True,
                frames_per_buffer   = self.chunksize)
        return self._stream

    def read(self, chunksize: Optional[int] = None) -> bytes:
        return cast(bytes, self.get_stream().read(self.chunksize, exception_on_overflow=False))

    def chunk_modify(self, chunk: bytes) -> bytes:
        if self.chosen_channel is not None and self.channel_count > 1:
            # filter out specific channel using numpy
            logging.info("Using numpy to filter out specific channel.")
            data = np.fromstring(chunk, dtype='int16').reshape((-1, self.channel_count))
            data = data[:, self.chosen_channel - 1]
            if watchdog:
                watchdog.sent_audio(data)
            chunk = data.tostring()
        return chunk

    def cleanup(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            #logger.debug("Stopped pyaudio stream")
            self._stream.close()
            #logger.debug("Closed pyaudio stream")

        if self._pyaudio is not None:
            self._pyaudio.terminate()
            #logger.debug("Terminated pyaudio")

    def set_input(self, id: int) -> None:
        devices = self.get_audio_devices()
        try:
            devName = devices[id]
            #logger.info(f'Using audio input device: {id} ({devName})')
            self.input_id = id
        except (ValueError, KeyError) as e:
            #logger.error(f'Unknown audio device: {id}')
            self.print_all_devices()
            sys.exit(1)

    def get_audio_devices(self) -> Dict[int, str]:
        devices = {}

        p = self.pyaudio
        info = p.get_host_api_info_by_index(0)
        deviceCount = info.get('deviceCount')

        for i in range(0, deviceCount):
                if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                        devices[i] = p.get_device_info_by_host_api_device_index(0, i).get('name')
        return devices

    def print_all_devices(self) -> None:
        """
        Special command, prints all audio devices available
        """
        print('-- AUDIO devices:')
        devices = self.get_audio_devices()
        for key in devices:
            dev = devices[key]
            if isinstance(dev, bytes):
                dev = dev.decode("ascii", "replace")
            print('    id=%i - %s' % (key, dev))

    @property
    def pyaudio(self) -> pyaudio.PyAudio:
        if self._pyaudio == None:
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio

    def set_audio_channel_filter(self, channel: int) -> None:
        # actually chosing a specific channel is apparently impossible with portaudio,
        # so we record all channels instead and then filter out the wanted channel with numpy
        if self.input_id is None:
            raise BugException()
        channelCount = self.pyaudio.get_device_info_by_host_api_device_index(0, self.input_id).get('maxInputChannels')
        #logger.info('Recording channel', channel, 'of', channelCount)
        self.channel_count = channelCount
        self.chosen_channel = channel

def segmenter_thread(stream_adapter, segmenter):
    while True:
        segmenter.append_signal(stream_adapter.read())

def processing_set_new_words(words=[]):
    model.new_words(words)

def insert_new_words(model, args, b=False):
    while True:
        if time.time() - model.time < 10: 
            words = []
            if "http" in args.new_words:
                for line in urllib.request.urlopen(args.new_words):
                    words.append(line.decode('utf-8').strip())
            else:
                with open(args.new_words,"r") as f:
                    for line in f:
                        line = line.strip()
                        words.append(line)

            model.new_words(words)

        if b:
            break
        time.sleep(1)

def decoding_thread(segmenter, model, punct, audiodevice):
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
                    cur_segs, wc = update_and_send(punct, device, dic, args.space, cur_segs, new_seg, wc, completed=False, audiodevice=audiodevice)
                else:
                    hypo = token2word(hypo, dic, args.space)
                    #print('Sending partial: ' + ' '.join(hypo))
                    send_hypo(start, end, hypo, args.outputType, final=False, audiodevice=audiodevice)

        if segment.completed:
            adc = segment.get_all()
            hypo, sp, ep, frames, best_memory_entry = decode(model, device, args, adc, fbank_mat, h_start, shypo)
            if len(hypo) == 0: continue
            start, end = ss + h_start*10, ss + (h_start+frames)*10

            hypo = hypo[1:-1]
            best_memory_entry = best_memory_entry[:-1]
            if punct is not None:
                new_seg = Segment(clean_noise(hypo, best_memory_entry, dic, args.space), start, end)
                cur_segs, wc = update_and_send(punct, device, dic, args.space, cur_segs, new_seg, wc, completed=True, audiodevice=audiodevice)
            else:
                hypo = token2word(hypo, dic, args.space)
                #print('Sending final: ' + ' '.join(hypo))
                send_hypo(start, end, hypo, args.outputType, final=True, audiodevice=audiodevice)
            
            segment.finish()
            #print('Finished segment.')
            
        time.sleep(0.2)

def send_hypo(start, end, hypo, output, final=False, audiodevice=None):
    #lh = 0 if output=='text' else len(hypo)
    #mcloud_w.send_packet_result_async(start, end, hypo, lh)
    if len(hypo) > 0:
        print(str(audiodevice)+" | "+str(start)+" | "+str(end)+" | "+(" ".join(hypo)))

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

def send_segs(segs, output, final=False, audiodevice=None):
    if len(segs) == 0: return
    start, end = segs[0].start, segs[-1].end
    hypo = []
    for seg in segs: hypo.extend(seg.text)
    hypo = ['<unk>' if w.startswith('%') or w.startswith('+') or w.startswith('<') or \
            w.startswith('*') or w.endswith('*') else w for w in hypo]
    #print('Sending %d %d %s' % (start, end, ' '.join(hypo)))
    send_hypo(start, end, hypo, output, final=final, audiodevice=audiodevice)

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


def update_and_send(punct, device, dic, space, segs, new_seg, wc, completed=False, audiodevice=None):
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
        #print('Sending final: ' + seg2text(old_segs))

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
        send_segs(old_segs, args.outputType, final=True, audiodevice=audiodevice)

    #if completed:
        #print(" ".join(token2word([x for s in new_segs for x in s.hypo],dic,space)))
    segs = punct_segs(punct, device, dic, space, new_segs)
    #if not completed:
        #print('Sending partial: ' + seg2text(segs))
    #else:
        #print('Sending final: ' + seg2text(segs))
    send_segs(segs, args.outputType, final=completed, audiodevice=audiodevice)

    punct.model.lock.release()

    return segs, wc

parser = argparse.ArgumentParser(description='pynn')
#model argument
parser.add_argument('--model-dic', help='model dictionary', default="model/s2s-lstm.dic")
parser.add_argument('--dict', help='dictionary file', default="model/bpe4k.dic")
parser.add_argument('--punct-dic', help='dictionary file', default="model/punct.dic")
parser.add_argument('--device', help='device', type=str, default='cpu')
parser.add_argument('--beam-size', help='beam size', type=int, default=4)
parser.add_argument('--attn-head', help='attention head', type=int, default=0)
parser.add_argument('--attn-padding', help='attention padding', type=float, default=0.05)
parser.add_argument('--stable-time', help='stable size', type=int, default=200)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--prune', help='pruning threshold', type=float, default=1.0)
parser.add_argument('--incl-block', help='incremental block size', type=int, default=50)
parser.add_argument('--max-len', help='max length', type=int, default=100)
parser.add_argument('--space', help='space token', type=str, default='▁')
parser.add_argument('--seg-based', help='output when audio segment is complete', action='store_true')
parser.add_argument('--new-words', help='path to text file with new words', default="words1.txt")

parser.add_argument('--chunksize', type=int, default=1024)

args = parser.parse_args()
args.outputType = "text"
#print(args)

fbank_mat = audio.filter_bank()
torch.set_grad_enabled(False)

stream_adapter = PortaudioStream(chunksize=args.chunksize)
stream_adapter.print_all_devices()

audiodevices = input("Select the audiodevice number (comma seperated list for multiple devices): ")

threads = []
for audiodevice in audiodevices.split(","):
    model, device, dic = init_asr_model(args)
    punct = init_punct_model(args)
    punct.model = model

    if args.new_words!="None":
        new_words = threading.Thread(target=insert_new_words, args=(model,args), daemon=True)
        new_words.start()
    else:
        processing_set_new_words()

    stream_adapter = PortaudioStream(chunksize=args.chunksize)
    stream_adapter.set_input(int(audiodevice))

    segmenter = Segmenter()
    
    record = threading.Thread(target=segmenter_thread, args=(stream_adapter, segmenter), daemon=True)
    threads.append(record)

    decoding = threading.Thread(target=decoding_thread, args=(segmenter, model, punct, audiodevice), daemon=True)
    threads.append(decoding)

for t in threads:
    t.start()
print("Models loaded and started.")

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Terminating running threads..")
