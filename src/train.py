# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:57:32 2016

@author: ORCHISAMA
"""

from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import lbg
from mel_coefficients import mfcc
from LPC import lpc
import matplotlib.pyplot as plt
import os


def training(nfiltbank, orderLPC):
    nSpeaker = 8
    nCentroid = 16
    codebooks_mfcc = np.empty((nSpeaker,nfiltbank,nCentroid))
    codebooks_lpc = np.empty((nSpeaker, orderLPC, nCentroid))
    directory = os.getcwd() + '/train';
    fname = str()

    for i in range(nSpeaker):
        fname = '/s' + str(i+1) + '.wav'
        print('Now speaker ', str(i+1), 'features are being trained' )
        (fs,s) = read(directory + fname)
        mel_coeff = mfcc(s, fs, nfiltbank)
        lpc_coeff = lpc(s, fs, orderLPC)
        codebooks_mfcc[i,:,:] = lbg(mel_coeff, nCentroid)
        codebooks_lpc[i,:,:] = lbg(lpc_coeff, nCentroid)
        
        plt.figure(i)
        plt.title('Codebook for speaker ' + str(i+1) + ' with ' + str(nCentroid) +  ' centroids')
        for j in range(nCentroid):
            plt.subplot(211)
            plt.stem(codebooks_mfcc[i,:,j])
            plt.ylabel('MFCC')
            plt.subplot(212)
            markerline, stemlines, baseline = plt.stem(codebooks_lpc[i,:,j])
            plt.setp(markerline,'markerfacecolor','r')
            plt.setp(baseline,'color', 'k')
            plt.ylabel('LPC')
            plt.axis(ymin = -1, ymax = 1)
            plt.xlabel('Number of features')
    
    plt.show()
    print('Training complete')
    
    #plotting 5th and 6th dimension MFCC features on a 2D plane
    #comment lines 54 to 71 if you don't want to see codebook
    codebooks = np.empty((2, nfiltbank, nCentroid))
    mel_coeff = np.empty((2, nfiltbank, 68))
   
    for i in range(2):
        fname = '/s' + str(i+2) + '.wav'
        (fs,s) = read(directory + fname)
        mel_coeff[i,:,:] = mfcc(s, fs, nfiltbank)[:,0:68]
        codebooks[i,:,:] = lbg(mel_coeff[i,:,:], nCentroid)
        
    
    plt.figure(nSpeaker + 1)
    s1 = plt.scatter(mel_coeff[0,6,:], mel_coeff[0,4,:],s = 100,  color = 'r', marker = 'o')
    c1 = plt.scatter(codebooks[0,6,:], codebooks[0,4,:], s = 100, color = 'r', marker = '+')
    s2 = plt.scatter(mel_coeff[1,6,:], mel_coeff[1,4,:],s = 100,  color = 'b', marker = 'o')
    c2 = plt.scatter(codebooks[1,6,:], codebooks[1,4,:], s = 100, color = 'b', marker = '+')
    plt.grid()
    plt.legend((s1, s2, c1, c2), ('Sp1','Sp2','Sp1 centroids', 'Sp2 centroids'), scatterpoints = 1, loc = 'upper left')    
    plt.show()
   
    
    return (codebooks_mfcc, codebooks_lpc)
    
    
