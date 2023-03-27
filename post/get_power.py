import numpy as np
import math
import fatpack
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import seaborn as sns
from scipy.signal import savgol_filter
import scipy.stats as stats   
import re

power = np.zeros((7,7,1,3))

for i in range(7):
    for j in range(7):
        for k in range(1):
            name = str(i*5)+"-"+str(j*5)+"-rotate-"+str(k)
            log_file = open("../job/"+name+"/log", "r")
            lines = log_file.readlines()
            last_lines = lines[-3]
            lst = re.findall(r'\d+', last_lines)
            power[i,j,k,:] = np.asarray([float(i) for i in lst])

# print(np.shape(root_moment))
np.save('power.npy', power)


total_power = np.sum(power,axis=3)

print(np.shape(total_power[:,:,0]))
plt.imshow(total_power[:,:,0]/total_power[0,0,0]-1)
plt.colorbar()
plt.savefig('power.png')
plt.colorbar()
print(total_power)