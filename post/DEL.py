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

caselist = ["ultralong-30","ultralong-20","ultralong-10","ultralong-0","ultralong+10","ultralong+20","ultralong+30"]
yaw_angle = [-30,-20,-10,0,10,20,30]
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
end = 399000
bins_num = 51
bins_max = 25
bins = np.linspace(0, bins_max, bins_num)
bin_width = bins_max/(bins_num-1)
bins_fine = np.linspace(0, bins_max, 501)


plt.figure(figsize=(14, 8),dpi=100)
plt.rcParams.update({'font.size': 22})

for ix,name in enumerate(caselist):
    print(ix)
    f[ix] = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
    time = np.array(f[ix].get('time'))[start:end]
    m_f[ix] = np.array(f[ix].get('moment_flap')[start:end,0,0,0])/1e6
    m_e[ix] = np.array(f[ix].get('moment_edge')[start:end,0,0,0])/1e6
    rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f[ix], h=1, k=256)
    ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
    ranges_corrected[ix] = Goodman_method_correction(ranges,means,np.max(m_f[ix]))
    N[ix], S[ix] = fatpack.find_range_count(ranges_corrected[ix], bins)
    DEL[ix] = (np.sum(N[ix]*S[ix]**m)/Neq)**(1/m)

DEL_test = DEL[ix_0]

for ii in range(3):
    for ix,name in enumerate(caselist):
        print(ix)
        f[ix] = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
        time = np.array(f[ix].get('time'))[start:end]
        m_f[ix] = np.array(f[ix].get('moment_flap')[start:end,0,0,ii])/1e6
        m_e[ix] = np.array(f[ix].get('moment_edge')[start:end,0,0,ii])/1e6
        rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f[ix], h=1, k=256)
        ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
        ranges_corrected[ix] = Goodman_method_correction(ranges,means,np.max(m_f[ix]))
        N[ix], S[ix] = fatpack.find_range_count(ranges_corrected[ix], bins)
        DEL[ix] = (np.sum(N[ix]*S[ix]**m)/Neq)**(1/m)


    plt.plot(yaw_angle,DEL/DEL_test,'o-',label="WT"+str(ii+1))

plt.legend()
plt.xlabel('Yaw angle (degree)')
plt.ylabel('$DEL/DEL_{baseline}$')
plt.savefig('plot/DEL.png')



df = pd.read_csv('power_3wt.csv')
print(df)


plt.figure(figsize=(14, 8),dpi=100)
plt.plot(df['yaw'],df['wt1']/4173025,'o-',label='WT1')
plt.plot(df['yaw'],df['wt2']/4173025,'o-',label='WT2')
plt.plot(df['yaw'],df['wt3']/4173025,'o-',label='WT3')
plt.legend()

plt.xlabel('Yaw angle (degree)')
plt.ylabel('$P/P_{baseline}$')
plt.savefig('plot/power.png')
