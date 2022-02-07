#!/usr/bin/python

import collections
import webrtcvad
import time

def frame_generator(frame_duration_ms, audio, sample_rate, start_chunk):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = start_chunk
    duration = (float(n) / sample_rate) / 2.0

    while offset + n <= len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += n
        offset += n

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

class AudioSegment():
    def __init__(self, start_time):
        self.time = start_time
        self.data = b''
        self.active = True
        self.completed = False

    def start_time(self):
        return self.time // (16*2)

    def end_time(self):
        return self.start_time() + self.size()

    def append(self, bytes):
        self.data += bytes

    def complete(self):
        self.completed = True

    def finish(self):
        self.active = False

    def get_all(self):
        return self.data

    def size(self):
        return len(self.data) // (16*2)

class Segmenter():
    def __init__(self, sample_rate=16000, VAD_aggressive=2, padding_duration_ms=450, frame_duration_ms=30, rate_begin=0.65, rate_end=0.55):

        self.vad = webrtcvad.Vad(VAD_aggressive)
        self.padding_duration_ms = padding_duration_ms
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        self.triggered = False
        self.length_of_frame = int(self.sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        self.rate_begin = rate_begin
        self.rate_end = rate_end

        self.segment = None
        self.segs = []
        self.ready = False 
        self.ring_buffer = collections.deque(maxlen=num_padding_frames)
        self.start_chunk = 0
        self.temp = b''

        self.reset()

    def reset(self):
        if self.segment is not None:
            self.segment.complete()
        self.segment = None

        self.triggered = False
        self.temp = b''
        self.start_chunk = 0
        self.ring_buffer.clear()
        self.segs = []
        self.ready = True

    def active_seg(self):
        for pos, seg in enumerate(self.segs):
            if seg.active: return (seg, pos)
        return (None, len(self.segs))

    def append_signal(self, audio):
        audio = self.temp + audio
        length_audio = len(audio) // self.length_of_frame * self.length_of_frame
        self.temp = audio[length_audio:]
        audio = audio[:length_audio]
        frames = frame_generator(self.frame_duration_ms, audio, self.sample_rate, self.start_chunk)
        frames = list(frames)
        self.start_chunk = self.start_chunk + len(audio)
        for j, frame in enumerate(frames):
            is_speech = self.vad.is_speech(frame.bytes, self.sample_rate)
            if not self.triggered:
                self.ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in self.ring_buffer if speech])

                if num_voiced >  self.rate_begin * self.ring_buffer.maxlen:
                    self.triggered = True
                    self.segment = AudioSegment(self.ring_buffer[0][0].timestamp)
                    for f, s in self.ring_buffer:
                        self.segment.append(f.bytes)
                    self.segs.append(self.segment)
                    #print('New segment added.')
                    self.ring_buffer.clear()
            else:
                self.segment.append(frame.bytes)
                self.ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                if num_unvoiced >  self.rate_end * self.ring_buffer.maxlen:
                    self.triggered = False
                    self.segment.complete()
                    self.ring_buffer.clear()













