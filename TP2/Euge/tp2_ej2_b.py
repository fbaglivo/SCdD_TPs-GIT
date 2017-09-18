# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:50:25 2017


H(x,y)=sum(sum(p(x,y)*log(p(x,y)/(log(x)*log(y)))))

@author: Fabri
"""

from scipy.io import loadmat
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


EPS = np.finfo(float).eps

"""
 Taken from https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
"""

def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi




if __name__ == "__main__":
  
    WAF = ['S01']
    
    epoch_time=0.8;
    

    
    for subject in range(len(WAF)):   
        
        data_dict= loadmat('matfiles/' + WAF[subject] + '.mat')
        data=data_dict['data'] # epoch,electrodes,time
        
        mi=np.zeros((data.shape[0],data.shape[1],data.shape[1]),dtype=float)
        
        for trial in  range(1): #range(data.shape[0]):

            for channel in range(data.shape[1]):
                
                for channel2 in range(channel,data.shape[1]):
            
                    mi[trial,channel,channel2]=mutual_information_2d(data[trial,channel,:],data[trial,channel2,:])        
            
                    print(channel, channel2)
                    
                    
    mi2=mi[0,:,:]    
    plt.imshow(mi2)   