# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:18:47 2017

@author: Fabri
"""


from scipy.io import loadmat
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 


if __name__ == "__main__":
    
    data_dict= loadmat('matfiles/S02.mat')
    data=data_dict['data'] # epoch,electrodes,time

    epoch_time=0.8;

    # Duración del epoch  1.54s
    # Baseline 200ms 

    # a.1) Calcular la media entre los electrodos 8, 44, 80, 131 y 185 
    #(el primer electrodo es el 0) y realizar una figura que muestre las 
    #frecuencias en el eje Y, los epochs en el eje X, y la potencia usando 
    #una escala de color


    channels=[7,43,79,130,184]   
    grand_average=np.mean(data[:,channels,:],1)
    
    Pxx = np.empty((101, len(grand_average)))
    
        
    for trial in range(len(grand_average)):    

        F,Pxx[:,trial]=welch(grand_average[trial,:],fs=grand_average.shape[1]/epoch_time, window='hamming')

   
    
    #plt.imshow(Pxx_log[:50,:],extent=[ 0, Pxx.shape[1],F[50], 0],aspect='auto')
    plt.imshow(Pxx[:50,:],extent=[ 0, Pxx.shape[1],F[50], 0],aspect='auto',cmap='viridis',norm=LogNorm(vmin=Pxx.min(), vmax=Pxx.max()))
         
           
    plt.colorbar()
    plt.xlabel('Epoch')
    plt.ylabel('frequency [Hz]')
    plt.yticks(F[:50])
    plt.show()    
    
    
    
        #a.2) Calcular la potencia media (entre epochs) para cada frecuencia
    # y graficar la potencia en funcion de la frecuencia para cada canal, como en el ejemplo:

    plt.figure()    
    for chanel in range(data.shape[1]): 

        for trial in range(data.shape[0]):
            F,Pxx[:,trial]=welch(data[trial,chanel,:],fs=data.shape[2]/epoch_time)
       
        Pxxm = np.mean(Pxx,axis=1)            
        plt.plot(F,Pxxm)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('V**/Hz')
