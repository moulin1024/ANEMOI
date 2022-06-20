import numpy as np
import math
import fatpack
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import seaborn as sns
from scipy.signal import savgol_filter
import scipy.stats as stats   

root_moment = np.zeros((13,13,11,3,2,49001))

for i in range(13):
    for j in range(13):
        for k in range(1):
            name = "rotate--2"+"-"+str((i-6)*5)+"-"+str((j-6)*5)
            print(name)
            f = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
            time = np.array(f.get('time'))
            # print(time.shape)
            for l in range(3):
                root_moment[i,j,k,l,0,:] = np.array(f.get('moment_flap')[:49001,0,0,l])/10**6
                root_moment[i,j,k,l,1,:] = np.array(f.get('moment_edge')[:49001,0,1,l])/10**6

# print(np.shape(root_moment))
np.save('root_moment_rotate_m2.npy', root_moment)
np.save('time.npy', time)
# caselist = ["0-0","20-10","20-20"]
# yaw_angle = [0,10,20]
# f = [None] * len(yaw_angle)
# m_f = [None] * len(yaw_angle)
# m_e = [None] * len(yaw_angle)
# N = [None] * len(yaw_angle)
# S = [None] * len(yaw_angle)
# ranges_corrected = [None] * len(yaw_angle)
# DEL = np.zeros(len(yaw_angle))
# ix_0 = len(yaw_angle)//2

# m = 10
# Neq = 1000
# start = 20000
# end = 100001
# bins_num = 31
# bins_max = 10
# bins = np.linspace(0, bins_max, bins_num)
# bin_width = bins_max/(bins_num-1)
# bins_fine = np.linspace(0, bins_max, 501)

# for ix,name in enumerate(caselist):
#     print(ix)
#     f[ix] = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
#     time = np.array(f[ix].get('time'))[start:]
#     m_f[ix] = np.array(f[ix].get('moment_flap')[start:end,0,0,2])/10**6
#     m_e[ix] = np.array(f[ix].get('moment_edge')[start:end,0,0,2])/10**6
#     rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f[ix], h=0.1, k=256)
#     ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
#     ranges_corrected[ix] = Goodman_method_correction(ranges,means,np.max(m_f[ix]))
#     N[ix], S[ix] = fatpack.find_range_count(ranges_corrected[ix], bins)
#     DEL[ix] = (np.sum(N[ix]*S[ix]**m)/Neq)**(1/m)
#     print(DEL[ix])

# fig, axs = plt.subplots(1, 3, figsize=(8, 5),dpi=200,sharex=True, sharey=True)
# for i in range(1):
#     for j in range(3):
#         case_ix = ix_0+(j-1)*(i+1)
#         print(case_ix)
#         axs[i].bar(S[case_ix], N[case_ix]/(np.sum(N[case_ix])*bin_width), width=bin_width,alpha=0.5,label="$\gamma=$"+str(yaw_angle[case_ix])+'$^\circ$')
#         # axs[i].set_ylim([0,0.4])
#         axs[i].legend()
#         axs[1].set_xlabel("Rainflow range ($mN \cdot m$)")
#         axs[0].set_ylabel("PDF")
# plt.savefig('range_histgram.png')
