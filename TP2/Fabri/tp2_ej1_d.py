# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:30:52 2017

@author: Fabri
"""


from scipy.io import loadmat
from scipy.signal import welch
#from scipy.linalg import logm
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def exact_mc_perm_test2(xs, ys, nmc):
    n = len(xs)
    k=[]
    diff = np.mean(xs) - np.mean(ys)
    
    for j in range(nmc):
        zs = np.concatenate([xs, ys])
        np.random.shuffle(zs)
        k.append(np.mean(zs[:n]) - np.mean(zs[n:]))
    if diff > 0:
        return float(np.sum(k>diff))/float(nmc)
    else:
        return float(np.sum(k<diff))/float(nmc)


def filtreate(F,banda):
    Filter1 =abs(F-banda.min())
    c1 = Filter1.min()
    inddase1 = np.where(Filter1==c1)[0]
    Filter2 =abs(F-banda.max())
    c2 = Filter2.min()
    inddase2 = np.where(Filter2==c2)[0]
    return inddase1[0],inddase2[0]

if __name__ == "__main__":


    WAF = ['P01','P02','P03','P04','P05','P06','P07','P08','P09','P10','S01','S02','S03','S04','S05','S06','S07','S08','S09','S10']

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
    del data_dict


    grupo = ['P'*int(len(WAF)/2),'S'*int(len(WAF)/2)]

    df_subj = pd.DataFrame({'Grupo' : grupo, 'Delta' : matrix2[:,0],'Thera' : matrix2[:,1],'Alpha' :matrix2[:,2], 'Beta' : matrix2[:,3],'Gamma' : matrix2[:,4]})


#d) Para cada banda de frecuencia, graficar según lo elegido en el punto c) y realizar un test estadístico apropiado.
    g = sns.factorplot(x="Grupo", y="Alpha", hue="Grupo",size=6, kind="violin", palette="muted",data=df_subj)
    g = sns.factorplot(x="Grupo", y="Beta", hue="Grupo",size=6, kind="violin", palette="muted",data=df_subj)
    g = sns.factorplot(x="Grupo", y="Gamma", hue="Grupo",size=6, kind="violin", palette="muted",data=df_subj)
    g = sns.factorplot(x="Grupo", y="Delta", hue="Grupo",size=6, kind="violin", palette="muted",data=df_subj)
    g = sns.factorplot(x="Grupo", y="Thera", hue="Grupo",size=6, kind="violin", palette="muted",data=df_subj)

# Permutaciones (solo porque no falla)
        
    df2 = df_subj.copy()

# Diferencias por bandas
#Alpha: 
    AlphaP = np.array(df2.Alpha[df2.Grupo == 'P'])
    AlphaS = np.array(df2.Alpha[df2.Grupo == 'S'])
    pperm1 = exact_mc_perm_test2(AlphaP,AlphaS, 1000) 
    print('Alpha P vs S: p=' + str(pperm1)+ ' (permutaciones).')   

#Beta: 
    BetaP = np.array(df2.Beta[df2.Grupo == 'P'])
    BetaS = np.array(df2.Beta[df2.Grupo == 'S'])
    pperm2 = exact_mc_perm_test2(BetaP,BetaS, 1000) 
    print('Beta P vs S: p=' + str(pperm2)+ ' (permutaciones).')  

#Gamma: 
    GammaP = np.array(df2.Gamma[df2.Grupo == 'P'])
    GammaS = np.array(df2.Gamma[df2.Grupo == 'S'])
    pperm3 = exact_mc_perm_test2(GammaP,GammaS, 1000) 
    print('Gamma P vs S: p=' + str(pperm3)+ ' (permutaciones).')  

#Delta: 
    DeltaP = np.array(df2.Delta[df2.Grupo == 'P'])
    DeltaS = np.array(df2.Delta[df2.Grupo == 'S'])
    pperm4 = exact_mc_perm_test2(DeltaP,DeltaS, 1000) 
    print('Delta P vs S: p=' + str(pperm4)+ ' (permutaciones).')  

#Thera: 
    TheraP = np.array(df2.Thera[df2.Grupo == 'P'])
    TheraS = np.array(df2.Thera[df2.Grupo == 'S'])
    pperm5 = exact_mc_perm_test2(AlphaP,AlphaS, 1000) 
    print('Thera P vs S: p=' + str(pperm5)+ ' (permutaciones).')  
