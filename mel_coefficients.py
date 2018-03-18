# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:48:15 2016

@author: ORCHISAMA
"""

from __future__ import division
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift, dct
import numpy as np
import matplotlib.pyplot as plt

def hertz_to_mel(freq):
    return 1125*np.log(1 + freq/700)
    
def mel_to_hertz(m):
    return 700*(np.exp(m/1125) - 1)
    
#calculate mel frequency filter bank
def mel_filterbank(nfft, nfiltbank, fs):
     
    #set limits of mel scale from 300Hz to 8000Hz
    lower_mel = hertz_to_mel(300)
    upper_mel = hertz_to_mel(8000)
    mel = np.linspace(lower_mel, upper_mel, nfiltbank+2)
    hertz = [mel_to_hertz(m) for m in mel]
    fbins = [int(hz * int(nfft/2+1)/fs) for hz in hertz]
    fbank = np.empty((int(nfft/2+1),nfiltbank))
    for i in range(1,nfiltbank+1):
        for k in range(int(nfft/2 + 1)):
            if k < fbins[i-1]:
                fbank[k, i-1] = 0
            elif k >= fbins[i-1] and k < fbins[i]:
                fbank[k,i-1] = (k - fbins[i-1])/(fbins[i] - fbins[i-1])
            elif k >= fbins[i] and k <= fbins[i+1]:
                fbank[k,i-1] = (fbins[i+1] - k)/(fbins[i+1] - fbins[i])
            else:
                fbank[k,i-1] = 0
     
#    plotting mel frequency filter banks           
#    plt.figure(1)
#    xbins = fs*np.arange(0,nfft/2+1)/(nfft/2+1)
#    for i in range(nfiltbank):
#        plt.plot(xbins, fbank[:,i])
#    plt.axis(xmax = 8000)
#    plt.xlabel('Frequency in Hz')
#    plt.ylabel('Amplitude')
#    plt.title('Mel Filterbank')
#    plt.show()
    return fbank
            
def mfcc(s,fs, nfiltbank):
  
    #divide into segments of 25 ms with overlap of 10ms
    nSamples = np.int32(0.025*fs)
    overlap = np.int32(0.01*fs)
    nFrames = np.int32(np.ceil(len(s)/(nSamples-overlap)))
    #zero padding to make signal length long enough to have nFrames
    padding = ((nSamples-overlap)*nFrames) - len(s)
    if padding > 0:
        signal = np.append(s, np.zeros(padding))
    else:
        signal = s
    segment = np.empty((nSamples, nFrames))
    start = 0
    for i in range(nFrames):
        segment[:,i] = signal[start:start+nSamples]
        start = (nSamples-overlap)*i
    
    #compute periodogram
    nfft = 512
    periodogram = np.empty((nFrames, int(nfft/2 + 1)))
    for i in range(nFrames):
        x = segment[:,i] * hamming(nSamples)
        spectrum = fftshift(fft(x,nfft))
        periodogram[i,:] = abs(spectrum[int(nfft/2-1):])/nSamples
        
    #calculating mfccs    
    fbank = mel_filterbank(nfft, nfiltbank, fs)
    #nfiltbank MFCCs for each frame
    mel_coeff = np.empty((nfiltbank,nFrames))
    for i in range(nfiltbank):
        for k in range(nFrames):
            mel_coeff[i,k] = np.sum(periodogram[k,:]*fbank[:,i])
            
    mel_coeff = np.log10(mel_coeff)
    mel_coeff = dct(mel_coeff)
    #exclude 0th order coefficient (much larger than others)
    mel_coeff[0,:]= np.zeros(nFrames)
    return mel_coeff
            

