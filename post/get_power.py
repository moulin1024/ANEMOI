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

power = np.zeros((13,13,5,3))
inflow = [-5,-3,0,3,5]
print(str(-2))
for i in range(13):
    print(i)
    for j in range(13):
        for k in range(5):
            name = "rotate-"+str(inflow[k])+"-"+str((i-6)*5)+"-"+str((j-6)*5)
            print(name)
            log_file = open("../job/"+name+"/log", "r")
            lines = log_file.readlines()
            last_lines = lines[-3]
            lst = re.findall(r'\d+', last_lines)
            print(last_lines,name)
            power[i,j,k,:] = np.asarray([float(i) for i in lst])

# # print(np.shape(root_moment))
np.save('mean_power.npy', power)


# total_power = np.sum(power,axis=3)

# print(np.shape(total_power[:,:,0]))
# plt.imshow(total_power[:,:,0])
# plt.colorbar()
# plt.savefig('power.png')
# plt.colorbar()
# print(total_power)
# power = np.zeros((13,13,49001,3))

# print(str(-2))
# for i in range(13):
#     # print(i)
#     for j in range(13):
#         for k in range(1):
#             name = "rotate-0-"+str((i-6)*5)+"-"+str((j-6)*5)
#             print(name)
#             data = pd.read_csv('../job/'+name+'/src/output/power.csv',header=None).to_numpy()
#             for l in range(3):
#                 power[i,j,:,l] = data[l::3,0]/1e6
#             # power[i,j,k,:]
#             # log_file = open("../job/"+name+"/log", "r")
#             # lines = log_file.readlines()
#             # last_lines = lines[-3]
#             # lst = re.findall(r'\d+', last_lines)
#             # print(last_lines,name)
#             # power[i,j,k,:] = np.asarray([float(i) for i in lst])

# mean_power = np.mean(power[:,:,10000:,:],axis=2)
# # # print(np.shape(root_moment))
# np.save('mean_power.npy', mean_power)


# total_power = np.sum(power,axis=3)

# print(np.shape(total_power[:,:,0]))
# plt.imshow(total_power[:,:,0])
# plt.colorbar()
# plt.savefig('power.png')
# plt.colorbar()
# print(total_power)