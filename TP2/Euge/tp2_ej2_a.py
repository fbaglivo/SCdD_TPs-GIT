# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:50:25 2017

@author: Fabri
"""

from scipy import stats
from scipy.io import loadmat
#from scipy.linalg import logm
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns

#c) Tomar la potencia de cada sujeto en la banda Alpha y graficar cada uno de los graficos categóricos de seaborn.
# {point, bar, count, box, violin, strip}


if __name__ == "__main__":


    WAF = ['S01']
    matrix2 = np.empty((len(WAF), 5))
    entropy = np.empty(len(WAF))    
    
    epoch_time=0.8;
    
    for subject in range(len(WAF)):   
        
        data_dict= loadmat('matfiles/' + WAF[subject] + '.mat')
        data=data_dict['data'] # epoch,electrodes,time
        entropy_per_subject = np.empty((data.shape[0],data.shape[1]))
    
        for trial in  range(data.shape[0]):

            for channel in range(data.shape[1]):
            
                zscored=stats.zscore(data[trial,channel,:])
                [n,bins]=np.histogram(zscored)
                entropy_per_subject[trial,channel]=stats.entropy(n)
                #print(channel)
        
            print(trial)
        
        entropy[subject]=np.mean(entropy_per_subject)