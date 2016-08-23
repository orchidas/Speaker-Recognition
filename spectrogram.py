# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 23:53:09 2016

@author: ORCHISAMA
"""
#calculate short time fourier transform and plot spectrogram
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
from scipy.signal import hann


def nearestPow2(inp):
    power = np.ceil(np.log2(inp))
    return 2**power

def stft(signal, fs, nfft, overlap):

    #plotting time domain signal
    plt.figure(1)
    t = np.arange(0,len(signal)/fs, 1/fs)
    plt.plot(t,signal)
    plt.axis(xmax = 1)
    plt.xlabel('Time in seconds')
    plt.ylabel('Amplitude')
    plt.title('Speech signal')
    
    if not np.log2(nfft).is_integer():
        nfft = nearestPow2(nfft)
    slength = len(signal)
    hop_size = np.int32(overlap * nfft)
    nFrames = int(np.round(len(signal)/(nfft-hop_size)))
    #zero padding to make signal length long enough to have nFrames
    signal = np.append(signal, np.zeros(nfft))
    STFT = np.empty((nfft, nFrames))
    segment = np.zeros(nfft)
    start = 0
    
    for n in range(nFrames):
        segment = signal[start:start+nfft] * hann(nfft) 
        padded_seg = np.append(segment,np.zeros(nfft))
        spec = fftshift(fft(padded_seg))
        spec = spec[len(spec)/2:]
        spec = abs(spec)/max(abs(spec))
        powerspec = 20*np.log10(spec)
        STFT[:,n] = powerspec
        start = start + nfft - hop_size 
        
    #plot spectrogram
    plt.figure(2)
    freq = (fs/(2*nfft)) * np.arange(0,nfft,1)
    time = np.arange(0,nFrames)*(slength/(fs*nFrames))
    plt.imshow(STFT, extent = [0,max(time),0,max(freq)],origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
    plt.ylabel('Frequency in Hz')
    plt.xlabel('Time in seconds')
    plt.axis([0,max(time),0,np.max(freq)])
    plt.title('Spectrogram of speech')
    plt.show()
    return (STFT, time, freq)
 
    


    
