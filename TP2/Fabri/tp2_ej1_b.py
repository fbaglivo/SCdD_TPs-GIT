# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:23:10 2017

@author: Fabri
"""

from scipy.io import loadmat
from scipy.signal import welch
#from scipy.linalg import logm
import numpy as np
#import matplotlib.pyplot as plt


#b) Calcular los valores de cada banda de frecuencia, promediados entre 
#   los electrodos (todos) y epochs para cada sujeto.

def filtreate(F,banda):
    Filter1 =abs(F-banda.min())
    c1 = Filter1.min()
    inddase1 = np.where(Filter1==c1)[0]
    Filter2 =abs(F-banda.max())
    c2 = Filter2.min()
    inddase2 = np.where(Filter2==c2)[0]
    return inddase1[0],inddase2[0]

if __name__ == "__main__":


    WAF = ['P01','S01']
    matrix2 = np.empty((len(WAF), 5))
    
    epoch_time=1.54;
    
    for subject in range(len(WAF)):   
        
        data_dict= loadmat('matfiles/' + WAF[subject] + '.mat')
        data=data_dict['data'] # epoch,electrodes,time
        
        grand_average_all=np.mean(data[:,:,:],1)
    
        FreQbins = 101
 
        Pxx = np.empty((FreQbins, len(grand_average_all)))
               
        for trial in range(len(grand_average_all)):    
    
            F,Pxx[:,trial]=welch(grand_average_all[trial,:],fs=len(grand_average_all[trial,:])/epoch_time)
            
        Pxx_r=np.reshape(Pxx,Pxx.shape[0]*Pxx.shape[1])
        Pxx_r=np.log10(Pxx_r)  
        Pxx=np.reshape(Pxx_r,(Pxx.shape[0],Pxx.shape[1]))

         
        #Delta < 4 Hz
        deltai,deltaf = filtreate(F,np.array([0,4]))
        #4 Hz <= Theta < 8 Hz
        therai, theraf =filtreate(F,np.array([4,8]))
        #8 Hz <= Alpha < 13 Hz
        alphai,alphaf = filtreate(F,np.array([8,13]))
        #13 Hz <= Beta < 30 Hz
        betai,betaf = filtreate(F,np.array([13,30]))
        #30 Hz <= Gamma < Nyquist
        gammai,gammaf = filtreate(F,np.array([30,45]))
        
        matrix = np.empty((data.shape[0], 5))
        matrix[:,0]= np.mean(Pxx[range(deltai,deltaf),:],axis=0)
        matrix[:,1]= np.mean(Pxx[range(therai, theraf),:],axis=0)
        matrix[:,2]= np.mean(Pxx[range(alphai,alphaf),:],axis=0)
        matrix[:,3]= np.mean(Pxx[range(betai,betaf),:],axis=0)
        matrix[:,4]= np.mean(Pxx[range(gammai,gammaf),:],axis=0)
        #df = pd.DataFrame(matrix,columns=list('dtabg'))
        
        
        matrix2[subject,:] = np.mean(matrix,axis=0)
        del matrix
 
    del data

   #grupo = ['P'*10+'S'*10]
    grupo = ['P','P','P','P','P','P','P','P','P','P','S','S','S','S','S','S','S','S','S','S']
    
    #df_subj = pd.DataFrame({'Grupo' : grupo, 'Delta' : matrix2[:,0],'Thera' : matrix2[:,1],'Alpha' :matrix2[:,2], 'Beta' : matrix2[:,3],'Gamma' : matrix2[:,4]})
    #