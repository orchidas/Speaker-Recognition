# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 00:34:54 2016

@author: ORCHISAMA
"""

#speaker specific Vector Quantization codebook using LBG algorithm
from __future__ import division
import numpy as np


def EUDistance(d,c):
    
    # np.shape(d)[0] = np.shape(c)[0]
    n = np.shape(d)[1]
    p = np.shape(c)[1]
    distance = np.empty((n,p))
    
    if n<p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:,i], (p,1)))
            distance[i,:] = np.sum((copies - c)**2,0)
    else:
        for i in range(p):
            copies = np.transpose(np.tile(c[:,i],(n,1)))
            distance[:,i] = np.transpose(np.sum((d - copies)**2,0))
            
    distance = np.sqrt(distance)
    return distance
            

def lbg(features, M):
    eps = 0.01
    codebook = np.mean(features, 1)
    distortion = 1
    nCentroid = 1
    while nCentroid < M:
        
        #double the size of codebook
        new_codebook = np.empty((len(codebook), nCentroid*2))
        if nCentroid == 1:
            new_codebook[:,0] = codebook*(1+eps)
            new_codebook[:,1] = codebook*(1-eps)
        else:    
            for i in range(nCentroid):
                new_codebook[:,2*i] = codebook[:,i] * (1+eps)
                new_codebook[:,2*i+1] = codebook[:,i] * (1-eps)
        
        codebook = new_codebook
        nCentroid = np.shape(codebook)[1]
        D = EUDistance(features, codebook)
        
        
        while np.abs(distortion) > eps:
       	    #nearest neighbour search
            prev_distance = np.mean(D)
            nearest_codebook = np.argmin(D,axis = 1)
            #print 'nearest neighbour',nearest_codebook
        
            #cluster vectors and find new centroid
            #print np.shape(np.mean(features[:, np.where(nearest_codebook == 0)],2))
            for i in range(nCentroid):
                codebook[:,i] = np.mean(features[:,np.where(nearest_codebook == i)], 2).T #add along 3rd dimension
          
            #replace all NaN values with 0  
            codebook = np.nan_to_num(codebook)    
            #print 'this codebook', codebook
            D = EUDistance(features, codebook)
            distortion = (prev_distance - np.mean(D))/prev_distance
            #print 'distortion' , distortion
            
    
    #print 'final codebook', codebook, np.shape(codebook)
    return codebook
        
            
