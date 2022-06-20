import numpy as np
import math
import fatpack
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import seaborn as sns
from scipy.signal import savgol_filter
import scipy.stats as stats   

# name = ["NREL-m-long","NREL-m-long-yaw-positive","NREL-m-long-yaw-negative"]
# power = np.zeros([299001,3,3])
# # print(name)
# for i in range(3):
#     f = h5py.File('../job/'+name[i]+'/output/'+name[i]+'_force.h5','r')
#     time = np.array(f.get('time'))

#     power[:,:,i] = np.array(f.get('power'))[:,0,0,:]
#     print(power.shape)

# plt.figure()
# plt.plot(power[:,0,0])
# plt.plot(power[:,1,0])
# plt.plot(power[:,2,0])
# plt.savefig('./power_history.png')


def Goodman_method_correction(M_a,M_m,M_max):
    M_u = 1.5*M_max
    M_ar = M_a/(1-M_m/M_u)
    return M_ar


name = ["NREL-m-long","NREL-m-long-yaw-positive","NREL-m-long-yaw-negative"]
# name = ["rotate-0-0-0","rotate-0-25-15","rotate-0--25--15"]
f = h5py.File('./data/'+name[0]+'_force.h5','r')

moment_flap = f['moment_flap'][()]

time = f['time'][()]/60

# ft = f['ft'][()]
# fx = f['fx'][()]
# phase = f['phase'][()][:,0,:,:]%(2*np.pi)

dr = 63/32
radius = np.linspace(dr,63,32)
# phase = np.squeeze(turb_force['phase'],axis=1)

print(moment_flap.shape)

m = 10
Neq = 1e6
bins_num = 51
bins_max = 15
bins = np.linspace(0, bins_max, bins_num)
bin_width = bins_max/(bins_num-1)

N_all = np.zeros([bins_num-1,3])
S_all = np.zeros([bins_num-1,3])
# end = 100000
time_window = np.linspace(5000,200000,39)*0.02/60
DEL_array = np.zeros(39)
for k in range(39):
    print(k)
    for i in range(3):
        m_f = moment_flap[:(5000+k*5000),0,i,0]/1e6
        rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=0.5, k=256)
        ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
        ranges_corrected = Goodman_method_correction(ranges,means,np.max(m_f))
        N, S = fatpack.find_range_count(ranges_corrected,bins)
        N_all[:,i] = N
        S_all[:,i] = S
        # DEL = (np.sum(N*S**m)/Neq)**(1/m)

    N_avg = np.mean(N_all,axis=1)
    S_avg = np.mean(S_all,axis=1)
    DEL_array[k] = (np.sum(N_avg*S_avg**m)/Neq)**(1/m)

fig, axs = plt.subplots(2,1,figsize=(12, 12), dpi=200)

# plt.rcParams['font.size'] = '20'
axs[0].plot(time,moment_flap[:,0,0,0]/1e6)
axs[0].plot([10,10],[0,12],'r--')
axs[0].set_xlim([0,time[200000]])
axs[0].set_ylim([0,12])
axs[0].set_ylabel('Bending moment ($mN \cdot m$)',fontsize=18)
axs[0].tick_params(axis='x', labelsize= 16)
axs[0].tick_params(axis='y', labelsize= 16)
axs[0].set_xlabel('Time (min)',fontsize=18)

axs[1].plot(time_window,DEL_array)
axs[1].plot([10,10],[-1,4],'r--',label='IEC recommendation')
axs[1].set_xlim([0,time[200000]])
axs[1].set_ylim([1,3])
axs[1].set_ylabel('DEL ($mN \cdot m$)',fontsize=18)
axs[1].set_xlabel('Time window size (min)',fontsize=18)
axs[1].tick_params(axis='x', labelsize= 16)
axs[1].tick_params(axis='y', labelsize= 16)
axs[1].legend(loc='lower right',fontsize=16)
# plt.rcParams.update({'font.size': 20})
plt.savefig('plot/robustness.png')

# plt.figure()
# for j in range(3):
#     for i in range(3):
#         m_f = moment_flap[:,0,i,j]/1e6
#         rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=1, k=256)
#         ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
#         ranges_corrected = Goodman_method_correction(ranges,means,np.max(m_f))
#         N, S = fatpack.find_range_count(ranges_corrected,bins)
#         N_all[:,i] = N
#         S_all[:,i] = S
#         # DEL = (np.sum(N*S**m)/Neq)**(1/m)

#     N_avg = np.mean(N_all,axis=1)
#     S_avg = np.mean(S_all,axis=1)
#     DEL_avg = (np.sum(N_avg*S_avg**m)/Neq)**(1/m)
#     print(DEL_avg)
#     # plt.bar(S,N)

#     plt.bar(np.mean(S_all,axis=1),np.mean(N_all,axis=1)*np.mean(S_all,axis=1)**m,width=bin_width,alpha=0.5)
# # plt.bar(S_all[:,1],N_all[:,1])
# # plt.bar(S_all[:,2],N_all[:,2])
# plt.savefig('plot/SN.png')

# print(DEL)
# y2 = np.sum(fx[:,0,:,0]*radius,axis=1)
# plt.figure()
# plt.plot(phase[:,0,0],y1,'.')
# plt.plot(phase[:,0,0],y2,'.')
# plt.plot(y2)
# plt.xlim([0,5000])
# fig, ax1 = plt.subplots()

# ax2 = ax1.twinx()
# ax1.plot(y1, 'g-')
# ax2.plot(y2, 'b-')

# ax1.set_xlabel('X data')
# ax1.set_ylabel('Y1 data', color='g')
# ax2.set_ylabel('Y2 data', color='b')
# ax1.set_xlim([8000,10000])
# ax2.set_xlim([8000,10000])

# plt.savefig('plot/fx.png')
# plt.figure()
# print(power.shape)
# plt.plot(phase[:,0,0],lw=1)
# plt.plot(phase[:,1,0],lw=1)
# plt.plot(phase[:,2,0],lw=1)
# plt.xlim([0,1000])
# # plt.plot(m_flap[:,0,0,1],lw=0.1,alpha=0.5)
# # plt.plot(m_flap[:,0,0,2],lw=0.1,alpha=0.5)
# # print(out_path)
# plt.savefig('plot/phase.png')
# dr = 63/32
# radius = np.linspace(dr/2,63-dr/2,32)
# # torque = np.sum(np.sum(np.swapaxes(ft,2,3)*radius,axis=3),axis=1)
# # rpm = power[:,0]/torque[:,0]

# from scipy.signal import savgol_filter

# test = np.vstack((phase[10000:,0,0],fx[10000:,0,24,0])).T
# test1 = np.vstack((phase[10000:,1,0],fx[10000:,1,24,0])).T
# test2 = np.vstack((phase[10000:,2,0],fx[10000:,2,24,0])).T
# data = np.vstack([test,test1,test2])
# data = data[data[:, 0].argsort()]
# fig = plt.figure()
# # plt.plot(phase[10000:30000,0,0],fx[10000:30000,0,20,0],'.')
# # plt.plot(phase[10000:30000,1,0],fx[10000:30000,1,20,0],'.')
# # plt.plot(phase[10000:30000,2,0],fx[10000:30000,2,20,0],'.')
# phase_all_blade =  data[:,0]
# force_all_blade =  data[:,1]
# yhat = savgol_filter(force_all_blade, 1001, 3) 
# plt.plot(phase_all_blade,yhat)
# plt.savefig('phase_force.png')

# # phase = np.cumsum(rpm)*0.02%(np.pi*2)
# # phase1 = (np.cumsum(rpm)*0.02+4/3*np.pi)%(np.pi*2)
# # phase2 = (np.cumsum(rpm)*0.02+2/3*np.pi)%(np.pi*2)

# # smoothed_data = np.zeros([100,32,3])
# # xnew = np.linspace(0.01,2*np.pi,100)

# from scipy.interpolate import interp1d

# smoothed_data = np.zeros([100,32,3])
# xnew = np.linspace(0.01,2*np.pi,100)

# for i in range(32):
#     print(i)
#     for j in range(3):
#         test = np.vstack((phase[0:10000,0,0],ft[0:10000,0,i,j])).T
#         test1 = np.vstack((phase[0:10000,1,0],ft[0:10000,1,i,j])).T
#         test2 = np.vstack((phase[0:10000,2,0],ft[0:10000,2,i,j])).T
#         data = np.vstack([test,test1,test2])
#         data = data[data[:, 0].argsort()]
#         yhat = savgol_filter(data[:,1], 101, 2) 
#         xnew = np.linspace(data[0,0],data[-1,0],100)
#         f = interp1d(data[:,0],yhat, kind='nearest')
#         smoothed_data[:,i,j] = f(xnew)

# # def phase_average(phase,force):
# #     smoothed_data = np.zeros([100,32,3])
# #     xnew = np.linspace(0.01,2*np.pi,100)
# #     for i in range(32):
# #         print(i)
# #         for j in range(3):
# #             test = np.vstack((phase[10000:,0,j],force[10000:,0,i,j])).T
# #             test1 = np.vstack((phase[10000:,1,j],force[10000:,1,i,j])).T
# #             test2 = np.vstack((phase[10000:,2,j],force[10000:,2,i,j])).T
# #             data = np.vstack([test,test1,test2])
# #             data = data[data[:, 0].argsort()]
# #             yhat = savgol_filter(data[:,1], 101,3) 
# #             xnew = np.linspace(data[0,0],data[-1,0],100)
# #             f = interp1d(data[:,0],yhat, kind='nearest')
# #             smoothed_data[:,i,j] = f(xnew)
# #     return smoothed_data


# # smooth_fx = phase_average(phase,ft)


# XX,YY = np.meshgrid(xnew,radius,indexing='ij')

# # fig = plt.figure(figsize=(18, 6), dpi=200)
# fig, axs = plt.subplots(1,3,figsize=(18, 8), dpi=200)
# plt.rcParams.update({'font.size': 18})
# for k in range(3):
#     axs[k].remove()
#     ax = fig.add_subplot(1,3,k+1,projection='polar')
#     im = ax.contourf(XX,YY,smoothed_data[:,:,k],40,vmin=0,vmax=1400)
#     # plt.colorbar(im1)
# cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.1])
# fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
# # ax[1] = fig.add_subplot(1,3,2,projection='polar')
# # ax[1].contourf(XX,YY,smoothed_data[:,:,1],40)

# # ax3 = fig.add_subplot(1,3,3,projection='polar')
# # ax3.contourf(XX,YY,smoothed_data[:,:,2],40)
# plt.savefig('phase_rotor.png')