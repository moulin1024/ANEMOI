import numpy as np
import math
import fatpack
# import rainflow
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import seaborn as sns
from scipy.signal import savgol_filter
import scipy.stats as stats   

def Goodman_method_correction(M_a,M_m,M_max):
    M_u = 1.5*M_max
    M_ar = M_a/(1-M_m/M_u)
    return M_ar

caselist = ["superlong-10","superlong-NREL-m","superlong+10"]
yaw_angle = [-10,0,10]
f = [None] * len(yaw_angle)
m_f = [None] * len(yaw_angle)
m_e = [None] * len(yaw_angle)
N = [None] * len(yaw_angle)
S = [None] * len(yaw_angle)
ranges_corrected = [None] * len(yaw_angle)
DEL = np.zeros(len(yaw_angle))
ix_0 = len(yaw_angle)//2

m = 10
Neq = 1000
start = 0
end = 80000
bins_num = 51
bins_max = 20
bins = np.linspace(0, bins_max, bins_num)
bin_width = bins_max/(bins_num-1)
bins_fine = np.linspace(0, bins_max, 501)

for ix,name in enumerate(caselist):
    print(ix)
    f[ix] = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
    time = np.array(f[ix].get('time'))[start:end]
    m_f[ix] = np.array(f[ix].get('moment_flap')[start:end,0,0,0])/1e3
    m_e[ix] = np.array(f[ix].get('moment_edge')[start:end,0,0,0])/1e3


fig, axs = plt.subplots(1, 1,figsize=(12, 4),dpi=200,sharex=True, sharey=True)
for i in range(3):
    # for j in range(3):
    # case_ix = ix_0
    # print(case_ix)
    plt.plot(time,m_f[i],label="$\gamma=$"+str(yaw_angle[i])+'$^\circ$',alpha=1,lw=1) 
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axs[0].set_ylim([4000,9000])
    # axs[0].set_xlabel('time (s)')
    # axs[0].set_ylabel('Flapwise bending moment ($mN \cdot m$)')
    # axs[i].rcParams.update({'font.size': 32})
plt.rcParams.update({'font.size': 32})

# for i in range(3):
#     for j in range(3):
#         case_ix = ix_0+(j-1)*(i+1)
#         axs[i].plot(time,m_f[case_ix],label="$\gamma=$"+str(yaw_angle[case_ix])+'$^\circ$',alpha=0.5) 
#         axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         # axs[i].set_ylim([1,14])
#         axs[2].set_xlabel('time (s)')
#         axs[1].set_ylabel('Flapwise bending moment ($mN \cdot m$)')
#         # axs[i].rcParams.update({'font.size': 32})
# # plt.rcParams.update({'font.size': 32})


plt.savefig('timeseries_flap.png')


# name = "ultralong-0"
# f1 = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
# time = np.array(f1.get('time'))[start:end]
# m_f1 = np.array(f1.get('moment_flap')[start:end,0,0,0])/1e6                                    
# m_f2 = np.array(f1.get('moment_flap')[start:end,0,0,1])/1e6
# m_f3 = np.array(f1.get('moment_flap')[start:end,0,0,2])/1e6

# fig, axs = plt.subplots(figsize=(12, 4),dpi=200)
# plt.plot(time,m_f1,alpha=0.5,label='WT1')
# plt.plot(time,m_f2,alpha=0.5,label='WT2')
# plt.plot(time,m_f3,alpha=0.5,label='WT3')
# plt.ylim([0,16])
# plt.xlabel('time (s)')
# plt.ylabel('Flapwise bending moment ($mN \cdot m$)')
# plt.legend()
# plt.savefig('plot/timeseries_3wt.png')

# name = "ultralong+20"
# f1 = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
# m_f1 = np.array(f1.get('moment_flap')[start:end,0,0,0])/1e6
# m_f2 = np.array(f1.get('moment_flap')[start:end,0,0,1])/1e6
# m_f3 = np.array(f1.get('moment_flap')[start:end,0,0,2])/1e6

# fig, axs = plt.subplots(figsize=(12, 4),dpi=200)
# plt.plot(time,m_f1,alpha=0.5,label='WT1')
# plt.plot(time,m_f2,alpha=0.5,label='WT2')
# plt.plot(time,m_f3,alpha=0.5,label='WT3')
# plt.ylim([0,16])
# plt.xlabel('time (s)')
# plt.ylabel('Flapwise bending moment ($mN \cdot m$)')
# plt.legend()
# plt.savefig('plot/timeseries_3wt+20.png')

# name = "ultralong-0"
# f1 = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
# time = np.array(f1.get('time'))[start:end]
# # m_f1 = np.array(f1.get('displacement_flap')[start:end,-3,0,0])

# # print(m_f1)
# # time = np.array(f1.get('time'))[start:]
# m_f1 = np.array(f1.get('moment_flap')[start:end,0,0,1])/1e3
# # m_f3 = np.array(f1.get('moment_flap')[start:end,0,0,2])/1e6

# fig, axs = plt.subplots(figsize=(12, 4),dpi=200)
# plt.plot(time-1000,m_f1,label='WT1')
# # plt.plot(time,m_f2,alpha=0.5,label='offset,zero yaw for WT1')
# # plt.plot(time,m_f3,alpha=0.5,label='offset,$-20^\circ$ yaw for WT1')
# plt.xlim([0,100])
# plt.ylim([2000,9000])

# plt.xlabel('time (s)')
# plt.ylabel('Flapwise bending moment ($kN \cdot m$)')
# plt.legend()
# plt.savefig('plot/timeseries_3wt_partial.png')



# fig, axs = plt.subplots(3, 1,figsize=(30, 10),dpi=100,sharex=True, sharey=True)
# plt.rcParams.update({'font.size': 22})
# for i in range(3):
#     for j in range(3):
#         case_ix = ix_0+(j-1)*(i+1)
#         axs[i].plot(time,m_e[case_ix],label="$\gamma=$"+str(yaw_angle[case_ix])+'$^\circ$',alpha=0.5) 
#         axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         # axs[i].set_ylim([1,14])
#         axs[i].set_xlim([1100,1200])
#         axs[2].set_xlabel('time (s)')
#         axs[1].set_ylabel('Flapwise bending moment ($mN \cdot m$)')
# plt.savefig('timeseries_edge.png')


# y1 = m_f1[start:end,0,0,0]/1e6#np.random.normal(size=1000) * 25.
# y2 = m_f2[start:end,0,0,0]/1e6#np.random.normal(size=1000) * 25.
# y3 = m_f3[start:end,0,0,0]/1e6#np.random.normal(size=1000) * 25.10
# time=time[start:end]

# rev1, ix1 = fatpack.find_reversals_racetrack_filtered(y1, h=1, k=256)
# rev2, ix2 = fatpack.find_reversals_racetrack_filtered(y2, h=1, k=256)
# rev3, ix3 = fatpack.find_reversals_racetrack_filtered(y3, h=1, k=256)

# plt.figure(figsize=(14, 8),dpi=100)
# plt.rcParams.update({'font.size': 22})
# plt.plot(time[ix1],y1[ix1], 'r.',alpha=0.4,label='reversal points')
# plt.plot(time,y1, lw=0.4)
# plt.legend()
# plt.xlabel('time (s)')
# plt.ylabel('flapwise bending moment ($mN \cdot m$)')
# plt.savefig('timeseries_baseline.png')


# plt.figure(figsize=(14, 8),dpi=100)
# plt.rcParams.update({'font.size': 22})
# plt.plot(time,y1, lw=1,alpha=0.8,label='Baseline')
# plt.plot(time,y2, lw=1,alpha=0.8,label='Positive 10 degree')
# plt.plot(time,y3, lw=1,alpha=0.8,label='Negative 10 degree')
# plt.legend()
# plt.ylim([0,18])
# plt.xlabel('time (s)')
# plt.ylabel('flapwise bending moment ($mN \cdot m$)')
# plt.savefig('timeseries_+-10.png')


# ranges1,means1 = fatpack.find_rainflow_ranges(rev1, k=256, return_means=True)
# ranges2,means2 = fatpack.find_rainflow_ranges(rev2, k=256, return_means=True)
# ranges3,means3 = fatpack.find_rainflow_ranges(rev3, k=256, return_means=True)

# ranges_corrected1 = Goodman_method_correction(ranges1,means1,np.max(y1))
# ranges_corrected2 = Goodman_method_correction(ranges2,means2,np.max(y2))
# ranges_corrected3 = Goodman_method_correction(ranges3,means3,np.max(y3))


# fit_a1,fit_c1,fit_loc1,fit_scale1=stats.exponweib.fit(ranges_corrected1,floc=0,fscale=1)
# fit_a2,fit_c2,fit_loc2,fit_scale2=stats.exponweib.fit(ranges_corrected2,floc=0,fscale=1)
# fit_a3,fit_c3,fit_loc3,fit_scale3=stats.exponweib.fit(ranges_corrected3,floc=0,fscale=1)

# # print(fit_a,fit_c)
# # # print(S.shape)
# # # Determine the fatigue damage, using a trilinear fatigue curve
# # # with detail category Sc, Miner's linear damage summation rule.
# # # Sc = 10.0
# # # curve = fatpack.TriLinearEnduranceCurve(Sc)
# # # fatigue_damage = curve.find_miner_sum(S)
# # # print(fatigue_damage)


# plt.figure(figsize=(14, 8),dpi=100)
# plt.rcParams.update({'font.size': 22})
# plt.plot(ranges_corrected1, '.',alpha=0.1)
# plt.plot(ranges_corrected2, '.',alpha=0.1)
# plt.plot(ranges_corrected3, '.',alpha=0.1)

# plt.gca().set_prop_cycle(None)

# # plt.gca().set_color_cycle(None)
# plt.plot(np.asarray(np.where(ranges_corrected1>8)).flatten(),ranges_corrected1[np.where(ranges_corrected1>8)], '.',label='Baseline')
# plt.plot(np.asarray(np.where(ranges_corrected2>8)).flatten(),ranges_corrected2[np.where(ranges_corrected2>8)], '.',label='Positive 10 degree')
# plt.plot(np.asarray(np.where(ranges_corrected3>8)).flatten(),ranges_corrected3[np.where(ranges_corrected3>8)], '.',label='Negative 10 degree')
# plt.legend()
# plt.xlabel('Cycle No.')
# plt.ylabel('Corrected Cycle range ($mN \cdot m$)')
# plt.savefig('range.png')
# # # plt.plot(ranges_corrected1, label='racetrack filtered reversals')

# # plt.plot(y2, alpha=.5)
# # plt.plot(y1)
# # plt.plot(y2)
# # # plt.plot(y3)
# # plt.legend(loc='best')
# # plt.xlabel("Index")
# # plt.ylabel("Signal")
# # # plt.xlim(30, 100)
# # plt.savefig('test2.png')

# bins_num = 51
# bins_max = 20
# bins = np.linspace(0, bins_max, bins_num)
# bin_width = bins_max/(bins_num-1)
# bins_fine = np.linspace(0, bins_max, 501)
# N_rtf1, S_rtf1 = fatpack.find_range_ix(ranges_corrected1, bins)
# N_rtf2, S_rtf2 = fatpack.find_range_ix(ranges_corrected2, bins)
# N_rtf3, S_rtf3 = fatpack.find_range_ix(ranges_corrected3, bins)

# DEL = (np.sum(N_rtf1*S_rtf1**m)/Neq)**(1/m)
# print(DEL)
# DEL = (np.sum(N_rtf2*S_rtf2**m)/Neq)**(1/m)
# print(DEL)
# DEL = (np.sum(N_rtf3*S_rtf3**m)/Neq)**(1/m)
# print(DEL)


# pdf1 = stats.exponweib.pdf(bins_fine, fit_a1,fit_c1,fit_loc1,fit_scale1)
# pdf2 = stats.exponweib.pdf(bins_fine, fit_a2,fit_c2,fit_loc2,fit_scale2)
# pdf3 = stats.exponweib.pdf(bins_fine, fit_a3,fit_c3,fit_loc3,fit_scale3)


# plt.figure(figsize=(12, 8),dpi=100)
# # plt.plot(bins_fine,y,label = 'Fitted Weibull: a='+str(round(fit_a,2))+',c='+str(round(fit_c,2)))

# plt.bar(S_rtf1, N_rtf1/(np.sum(N_rtf1)*bin_width),alpha=0.5, width=bin_width,label='Baseline')
# plt.bar(S_rtf2, N_rtf2/(np.sum(N_rtf2)*bin_width),alpha=0.5, width=bin_width,label='Positive 20 degree')
# plt.bar(S_rtf3, N_rtf3/(np.sum(N_rtf3)*bin_width),alpha=0.5, width=bin_width,label='Negative 20 degree')
# plt.ylim([0,0.4])
# plt.legend()
# plt.xlabel("Rainflow range ($mN \cdot m$)")
# plt.ylabel("PDF")
# plt.savefig('range_histgram.png')



# plt.figure(figsize=(12, 8),dpi=100)
# plt.bar(S_rtf1, np.cumsum(S_rtf1**m*N_rtf1),alpha=0.5, width=bin_width)
# plt.bar(S_rtf2, np.cumsum(S_rtf2**m*N_rtf2),alpha=0.5, width=bin_width)
# plt.bar(S_rtf3, np.cumsum(S_rtf3**m*N_rtf3),alpha=0.5, width=bin_width)
# plt.legend()
# plt.xlabel("Rainflow range ($mN \cdot m$)")
# plt.ylabel("Contribution to the 10th moment of the PDF")
# plt.savefig('range_histgram_mth.png')


