import math
import numpy as np
import scipy.io.wavfile

def filter_bank(sample_rate, nfft, filters):    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, filters + 2, dtype="float32")  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((filters, int(np.floor(nfft / 2 + 1))), dtype="float32")
    for m in range(1, filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank.T

def signal_to_fft(signal, sample_rate, nfft):
    signal = signal.astype("float32")
    pre_emp = 0.97 # Pre-emphasis
    emp_signal = np.append(signal[0], signal[1:] - pre_emp*signal[:-1])
    frame_size = 0.025 # frame size of 25ms
    frame_stride = 0.01 # frame move of 10ms

    # Convert from seconds to samples
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate 
    signal_length = len(emp_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    
    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  
    
    pad_signal_length = num_frames * frame_step + frame_length
    
    z = np.zeros((pad_signal_length - signal_length), dtype="float32")
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(emp_signal, z) 

    indices = np.tile(np.arange(0, frame_length),
            (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    frames *= np.hamming(frame_length)
    # Explicit Implementation
    # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))

    # Magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(frames, nfft), dtype="float32")
    # Power Spectrum
    pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))

    return pow_frames

def extract_fft(wav, start, end, nfft=256):
    sample_rate, signal = scipy.io.wavfile.read(wav)
    pow_frames = signal_to_fft(signal, sample_rate, nfft)
    return np.log10(pow_frames)
    
def extract_fbank(signal, fbank_mat=None, sample_rate=16000, nfft=256, filters=40):
    pow_frames = signal_to_fft(signal, sample_rate, nfft)

    if fbank_mat is None:
        fbank_mat = filter_bank(sample_rate, nfft, filters)
    
    fbanks = np.dot(pow_frames, fbank_mat)
    # Numerical Stability
    fbanks = np.where(fbanks==0, np.finfo(float).eps, fbanks)
    fbanks = np.log10(fbanks)  # dB
    
    return fbanks

