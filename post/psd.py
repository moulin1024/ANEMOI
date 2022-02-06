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

# caselist = ["ultralong-30","ultralong-20","ultralong-10","ultralong-0","ultralong+10","ultralong+20","ultralong+30"]
# yaw_angle = [-30,-20,-10,0,10,20,30]
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
start = 39000
bins_num = 51
bins_max = 20
bins = np.linspace(0, bins_max, bins_num)
bin_width = bins_max/(bins_num-1)
bins_fine = np.linspace(0, bins_max, 501)

name = 'superlong-NREL-m'
plt.plot(m_f)
plt.savefig('test.png')
# for ix,name in enumerate(caselist):
#     print(ix)
#     f[ix] = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
#     time = np.array(f[ix].get('time'))[start:]
#     m_f[ix] = np.array(f[ix].get('moment_flap')[start:,0,0,0])/1e6
#     m_e[ix] = np.array(f[ix].get('moment_edge')[start:,0,0,0])/1e6
#     rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f[ix], h=1, k=256)
#     ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
#     ranges_corrected[ix] = Goodman_method_correction(ranges,means,np.max(m_f[ix]))
#     N[ix], S[ix] = fatpack.find_range_count(ranges_corrected[ix], bins)
#     DEL[ix] = (np.sum(N[ix]*S[ix]**m)/Neq)**(1/m)

f = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')

fig = plt.figure(figsize=(10, 8),dpi=100)
plt.rcParams.update({'font.size': 22})

m_f = np.array(f.get('fx')[:,0,26,0])
data = m_f-np.mean(m_f[ix_0])
ps = np.abs(np.fft.fft(data))**2
time_step = 0.02
freqs = np.fft.fftfreq(data.size, time_step)
idx = np.argsort(freqs)
plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='Flapwise')

m_f = np.array(f.get('moment_flap')[:,0,26,0])
data = m_f-np.mean(m_f[ix_0])
ps = np.abs(np.fft.fft(data))**2
time_step = 0.02
freqs = np.fft.fftfreq(data.size, time_step)
idx = np.argsort(freqs)
plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='Flapwise')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectrum density')
plt.legend()
# plt.ylim([1e-6,1e10])
plt.savefig('plot/psd_baseline.png')


# fig = plt.figure(figsize=(10, 8),dpi=100)
# plt.rcParams.update({'font.size': 22})
# data = m_f[2]-np.mean(m_f[2])
# ps = np.abs(np.fft.fft(data))**2
# time_step = 0.02
# freqs = np.fft.fftfreq(data.size, time_step)
# idx = np.argsort(freqs)
# plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='$\gamma = -10^\circ$')

# data = m_f[4]-np.mean(m_f[4])
# ps = np.abs(np.fft.fft(data))**2
# plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='$\gamma = 10^\circ$')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power spectrum density')
# plt.legend()
# plt.ylim([1e-6,1e10])
# plt.savefig('plot/psd_+-10.png')

# fig = plt.figure(figsize=(10, 8),dpi=100)
# plt.rcParams.update({'font.size': 22})
# data = m_f[1]-np.mean(m_f[1])
# ps = np.abs(np.fft.fft(data))**2
# time_step = 0.02
# freqs = np.fft.fftfreq(data.size, time_step)
# idx = np.argsort(freqs)
# plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='$\gamma = -20^\circ$')

# data = m_f[5]-np.mean(m_f[5])
# ps = np.abs(np.fft.fft(data))**2
# plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='$\gamma = 20^\circ$')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power spectrum density')
# plt.legend()
# plt.ylim([1e-6,1e10])
# plt.savefig('plot/psd_+-20.png')


# fig = plt.figure(figsize=(10, 8),dpi=100)
# plt.rcParams.update({'font.size': 22})
# data = m_f[0]-np.mean(m_f[0])
# ps = np.abs(np.fft.fft(data))**2
# time_step = 0.02
# freqs = np.fft.fftfreq(data.size, time_step)
# idx = np.argsort(freqs)
# plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='$\gamma = -30^\circ$')

# data = m_f[6]-np.mean(m_f[6])
# ps = np.abs(np.fft.fft(data))**2
# plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='$\gamma = 30^\circ$')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power spectrum density')
# plt.legend()
# plt.ylim([1e-6,1e10])
# plt.savefig('plot/psd_+-30.png')
