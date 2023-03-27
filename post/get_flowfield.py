import numpy as np
import math
import fatpack
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import seaborn as sns
from scipy.signal import savgol_filter
import scipy.stats as stats   

flowfield = np.zeros((4,4,6,6,512,256))

for i in range(4):
    for j in range(4):
        for k in range(6):
            name = str(i*10)+"-"+str(j*10)+"-rotate-"+str(k)
            f = h5py.File('../job/'+name+'/output/'+name+'_stat.h5','r')
            flowfield[i,j,k,0,:,:] = np.array(f.get('u_avg')[:,:,44])
            flowfield[i,j,k,1,:,:] = np.array(f.get('v_avg')[:,:,44])
            flowfield[i,j,k,2,:,:] = np.array(f.get('w_avg')[:,:,44])
            flowfield[i,j,k,3,:,:] = np.array(f.get('u_std')[:,:,44])
            flowfield[i,j,k,4,:,:] = np.array(f.get('v_std')[:,:,44])
            flowfield[i,j,k,5,:,:] = np.array(f.get('w_std')[:,:,44])

np.save('flowfield.npy', flowfield)