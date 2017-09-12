# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:18:47 2017

@author: Fabri
"""


from scipy.io import loadmat
from scipy.signal import welch
from scipy.linalg import logm
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    data_dict= loadmat('matfiles/S02.mat')
    data=data_dict['data'] # epoch,electrodes,time

    epoch_time=1.54;

    # Duración del epoch  1.54s
    # Baseline 200ms 

    # a.1) Calcular la media entre los electrodos 8, 44, 80, 131 y 185 
    #(el primer electrodo es el 0) y realizar una figura que muestre las 
    #frecuencias en el eje Y, los epochs en el eje X, y la potencia usando 
    #una escala de color


    channels=[7,43,79,130,145]   
    grand_average=np.mean(data[:,channels,:],1)
    
    Pxx = np.empty((101, len(grand_average)))
    
        
    for trial in range(len(grand_average)):    

        F,Pxx[:,trial]=welch(grand_average[trial,:],fs=130,nperseg=130, noverlap=32,window='hamming')

    Pxx_r=np.reshape(Pxx,Pxx.shape[0]*Pxx.shape[1])
    Pxx_r=np.log10(Pxx_r)  
    Pxx_log=np.reshape(Pxx_r,(Pxx.shape[0],Pxx.shape[1]))
    
    
    plt.imshow(Pxx_log[:30,:],extent=[ 0, Pxx.shape[1],F[30], 0],aspect='auto')
                
    plt.colorbar()
    plt.xlabel('Epoch')
    plt.ylabel('frequency [Hz]')
    plt.yticks(F[:30])
    plt.show()    