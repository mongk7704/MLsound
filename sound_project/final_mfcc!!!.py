
# coding: utf-8

# In[2]:

import numpy as np
import math as m
class mfcc_param:
    def _init_(self):
     ("")
def myRound(n):
    import math as mt;
    a=int(n)
    if a%2==0:
        return mt.ceil(n)
    else:
        return round(n)

def mfcc_init(window,fs,FFT):  
    mfccParams=mfcc_param()
    mfccParams.cepstralCoefficients = 13;
    mfccParams.fftSize = myRound(window / 2);
    mfccParams.lowestFrequency = 133.3333;
    mfccParams.linearFilters = 13;
    mfccParams.linearSpacing = 66.66666666;
    mfccParams.logFilters = 27;
    mfccParams.logSpacing = 1.0711703;
    mfccParams.totalFilters = mfccParams.linearFilters + mfccParams.logFilters;
    mfccParams.freqs =  mfccParams.lowestFrequency +np.multiply(list(range(0,mfccParams.linearFilters)),mfccParams.linearSpacing);
    f=mfccParams.freqs[mfccParams.linearFilters-1] * np.power(mfccParams.logSpacing,np.array(list(range(1,mfccParams.logFilters+3))));
    mfccParams.freqs=np.hstack((mfccParams.freqs, f));

    mfccParams.lower = mfccParams.freqs[0:mfccParams.totalFilters];
    mfccParams.center = mfccParams.freqs[1:mfccParams.totalFilters+1];
    mfccParams.upper = mfccParams.freqs[2:mfccParams.totalFilters+2];
    mfccParams.mfccFilterWeights = np.zeros((mfccParams.totalFilters,mfccParams.fftSize));
#mfccParams.freqs()
   
    mfccParams.triangleHeight = np.divide(2,(mfccParams.upper-mfccParams.lower));
    mfccParams.fftFreqs = np.multiply(np.divide(list(range(0,mfccParams.fftSize)),mfccParams.fftSize),fs);
###
    for chan in range(0,mfccParams.totalFilters): 
        a=(mfccParams.fftFreqs > mfccParams.lower[chan]);
        b=(mfccParams.fftFreqs<=mfccParams.center[chan]);
        ab=np.logical_and(a,b);
        ab=ab*mfccParams.triangleHeight[chan]*(mfccParams.fftFreqs-mfccParams.lower[chan]);
        ab=ab/(mfccParams.center[chan]-mfccParams.lower[chan]);
        c=(mfccParams.fftFreqs > mfccParams.center[chan]);
        d=(mfccParams.fftFreqs < mfccParams.upper[chan]);
        cd=np.logical_and(c,d)*mfccParams.triangleHeight[chan];
        cd=cd*(mfccParams.upper[chan]-mfccParams.fftFreqs)/(mfccParams.upper[chan]-mfccParams.center[chan]);
        abcd=ab+cd;
        mfccParams.mfccFilterWeights[chan,:]=abcd;
    
    mfccParams.mfccDCTMatrix = np.multiply(1/np.sqrt(mfccParams.totalFilters/2)  ,np.cos(np.multiply(np.matmul( np.reshape(list(range(0,(mfccParams.cepstralCoefficients))),(13,1) ), np.transpose(np.reshape(np.add(np.multiply(2,list(range(0,(mfccParams.totalFilters)))),1),(40,1))) )
                       ,(m.pi/2/mfccParams.totalFilters)))) 

                        
    mfccParams.mfccDCTMatrix[0,:] = np.divide(np.multiply(mfccParams.mfccDCTMatrix[0,:] , np.sqrt(2)),2);
    
    earMag = np.log10(np.add(np.matmul(mfccParams.mfccFilterWeights , FFT),np.finfo(float).eps));
    ceps = np.matmul(mfccParams.mfccDCTMatrix , earMag);
    return ceps;

