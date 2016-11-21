
# coding: utf-8

# In[1]:


import numpy as np;
import math as math;
from numpy.fft import fft,fftshift;
import matplotlib.pyplot as plt;
from mfcc_param import*
import scipy.io as sio;

from svmutil import *;
from svm import *;

from scipy.io import wavfile # get the api
def positiveFFT(Fs,data):
        
        x=data
        N=len(x); 
        k=list(range(0,N)); 
        T=N/Fs;      
        freq=np.divide(k,T);
        X=fft(x)/N*2;
        cutOff = math.ceil(N/2);
        X = X[0:cutOff];
        freq = freq[0:cutOff];
        return X,freq
def data_input(name):
        Fs, y = wavfile.read(name)
        m=len(np.shape(y))
        if m!=1:
            y=y[:,0];
        a,freq=positiveFFT(Fs,y);
        x=abs(a);
        x_pow=pow(x,2);
        i=np.where(x==max(x));
        mfcc=mfcc_init(len(y),Fs,x);
        #print(type(i))
        #np.g
        #mfccParam=feature_mfccs_init(length(y),Fs);
        #mfcc=feature_mfccs(abs(a),mfccParam);
        #feature1=extractFeature(abs(a),b');      
        #%feature=[mfcc',feature1];
        #feature=feature1;
        feature=[[np.mean(x),np.mean(x_pow),np.var(x),np.max(x),np.min(x),freq[i[0]][0],np.max(freq),np.min(freq)]];
        for x in mfcc.tolist():
            feature[0].append(x);
        return feature;
def test_data():
    get_ipython().magic('cd sound')
    e=data_input('01.wav')
    e=np.vstack((e,data_input('02.wav')))
    e=np.vstack((e,data_input('03.wav')))
    e=np.vstack((e,data_input('04.wav')))
    e=np.vstack((e,data_input('05.wav')))
    e=np.vstack((e,data_input('06.wav')))
    e=np.vstack((e,data_input('07.wav')))
    e=np.vstack((e,data_input('08.wav')))
    e=np.vstack((e,data_input('09.wav')))
    e=np.vstack((e,data_input('10.wav')))
    e=np.vstack((e,data_input('11.wav')))
    e=np.vstack((e,data_input('12.wav')))
    e=np.vstack((e,data_input('13.wav')))
    e=np.vstack((e,data_input('14.wav')))
    e=np.vstack((e,data_input('15.wav')))
    
        
    
    e=np.vstack((e,data_input('51.wav')))
    e=np.vstack((e,data_input('52.wav')))
    e=np.vstack((e,data_input('53.wav')))
    e=np.vstack((e,data_input('54.wav')))
    e=np.vstack((e,data_input('55.wav')))
    e=np.vstack((e,data_input('56.wav')))
    e=np.vstack((e,data_input('57.wav')))
    e=np.vstack((e,data_input('58.wav')))
    e=np.vstack((e,data_input('59.wav')))
    e=np.vstack((e,data_input('60.wav')))
    e=np.vstack((e,data_input('61.wav')))
    e=np.vstack((e,data_input('62.wav')))
    e=np.vstack((e,data_input('63.wav')))
    e=np.vstack((e,data_input('64.wav')))
    e=np.vstack((e,data_input('65.wav')))
    
    
    
    e=np.vstack((e,data_input('201.wav')))
    e=np.vstack((e,data_input('202.wav')))
    e=np.vstack((e,data_input('203.wav')))
    e=np.vstack((e,data_input('204.wav')))
    e=np.vstack((e,data_input('205.wav')))
    e=np.vstack((e,data_input('206.wav')))
    e=np.vstack((e,data_input('207.wav')))
    e=np.vstack((e,data_input('208.wav')))
    e=np.vstack((e,data_input('209.wav')))
    e=np.vstack((e,data_input('210.wav')))
    e=np.vstack((e,data_input('211.wav')))
    e=np.vstack((e,data_input('212.wav')))
    e=np.vstack((e,data_input('213.wav')))
    e=np.vstack((e,data_input('214.wav')))
    e=np.vstack((e,data_input('215.wav')))
   
    return e

