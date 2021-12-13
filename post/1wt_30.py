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

def get_DEL(y,Neq,m):
    S, Sm = fatpack.find_rainflow_ranges(y.flatten(), return_means=True, k=256)
    M_ar = Goodman_method_correction(S,Sm,np.max(y))
    hist, bin_edges = np.histogram(M_ar,bins=51)
    bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
    DEL = (np.sum(hist*bin_centres**m)/Neq)**(1/m)
    return DEL

case1 = "superlong-NREL-m"
case2 = "superlong+30"
case3 = "superlong-30"

f1 = h5py.File('../job/'+case1+'/output/'+case1+'_force.h5','r')
f2 = h5py.File('../job/'+case2+'/output/'+case2+'_force.h5','r')
f3 = h5py.File('../job/'+case3+'/output/'+case3+'_force.h5','r')

m_f1 = np.array(f1.get('moment_flap'))
m_f2 = np.array(f2.get('moment_flap'))
m_f3 = np.array(f3.get('moment_flap'))

time = np.array(f3.get('time'))
print(time.shape)

m = 10
Neq = 1000
start = 19000
end = 159000

y1 = m_f1[start:,0,0,0]/1e6#np.random.normal(size=1000) * 25.
y2 = m_f2[start:,0,0,0]/1e6#np.random.normal(size=1000) * 25.
y3 = m_f3[start:,0,0,0]/1e6#np.random.normal(size=1000) * 25.10
time=time[start:]

rev_rtf1, ix_rtf1 = fatpack.find_reversals_racetrack_filtered(y1, h=1, k=256)
rev_rtf2, ix_rtf2 = fatpack.find_reversals_racetrack_filtered(y2, h=1, k=256)
rev_rtf3, ix_rtf3 = fatpack.find_reversals_racetrack_filtered(y3, h=1, k=256)

plt.figure(figsize=(14, 8),dpi=100)
plt.rcParams.update({'font.size': 22})
plt.plot(time[ix_rtf1],y1[ix_rtf1], 'r.',alpha=0.4,label='reversal points')
plt.plot(time,y1, lw=0.4)
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('flapwise bending moment ($mN \cdot m$)')
plt.savefig('timeseries_baseline.png')


plt.figure(figsize=(14, 8),dpi=100)
plt.rcParams.update({'font.size': 22})
plt.plot(time,y1, lw=1,alpha=0.8,label='Baseline')
plt.plot(time,y2, lw=1,alpha=0.8,label='Positive 30 degree')
plt.plot(time,y3, lw=1,alpha=0.8,label='Negative 30 degree')
plt.legend()
plt.ylim([0,18])
plt.xlabel('time (s)')
plt.ylabel('flapwise bending moment ($mN \cdot m$)')
plt.savefig('timeseries_+-30.png')


ranges_rtf1,means_rtf1 = fatpack.find_rainflow_ranges(rev_rtf1, k=256, return_means=True)
ranges_rtf2,means_rtf2 = fatpack.find_rainflow_ranges(rev_rtf2, k=256, return_means=True)
ranges_rtf3,means_rtf3 = fatpack.find_rainflow_ranges(rev_rtf3, k=256, return_means=True)

ranges_corrected_rtf1 = Goodman_method_correction(ranges_rtf1,means_rtf1,np.max(y1))
ranges_corrected_rtf2 = Goodman_method_correction(ranges_rtf2,means_rtf2,np.max(y2))
ranges_corrected_rtf3 = Goodman_method_correction(ranges_rtf3,means_rtf3,np.max(y3))


fit_a1,fit_c1,fit_loc1,fit_scale1=stats.exponweib.fit(ranges_corrected_rtf1,floc=0,fscale=1)
fit_a2,fit_c2,fit_loc2,fit_scale2=stats.exponweib.fit(ranges_corrected_rtf2,floc=0,fscale=1)
fit_a3,fit_c3,fit_loc3,fit_scale3=stats.exponweib.fit(ranges_corrected_rtf3,floc=0,fscale=1)

# print(fit_a,fit_c)
# # print(S.shape)
# # Determine the fatigue damage, using a trilinear fatigue curve
# # with detail category Sc, Miner's linear damage summation rule.
# # Sc = 10.0
# # curve = fatpack.TriLinearEnduranceCurve(Sc)
# # fatigue_damage = curve.find_miner_sum(S)
# # print(fatigue_damage)


plt.figure(figsize=(14, 8),dpi=100)
plt.rcParams.update({'font.size': 22})
plt.plot(ranges_corrected_rtf1, '.',alpha=0.1)
plt.plot(ranges_corrected_rtf2, '.',alpha=0.1)
plt.plot(ranges_corrected_rtf3, '.',alpha=0.1)

plt.gca().set_prop_cycle(None)

# plt.gca().set_color_cycle(None)
plt.plot(np.asarray(np.where(ranges_corrected_rtf1>8)).flatten(),ranges_corrected_rtf1[np.where(ranges_corrected_rtf1>8)], '.',label='Baseline')
plt.plot(np.asarray(np.where(ranges_corrected_rtf2>8)).flatten(),ranges_corrected_rtf2[np.where(ranges_corrected_rtf2>8)], '.',label='Positive 30 degree')
plt.plot(np.asarray(np.where(ranges_corrected_rtf3>8)).flatten(),ranges_corrected_rtf3[np.where(ranges_corrected_rtf3>8)], '.',label='Negative 30 degree')
plt.legend()
plt.xlabel('Cycle No.')
plt.ylabel('Corrected Cycle range ($mN \cdot m$)')
plt.savefig('range_30.png')
# # plt.plot(ranges_corrected_rtf1, label='racetrack filtered reversals')

# plt.plot(y2, alpha=.5)
# plt.plot(y1)
# plt.plot(y2)
# # plt.plot(y3)
# plt.legend(loc='best')
# plt.xlabel("Index")
# plt.ylabel("Signal")
# # plt.xlim(30, 100)
# plt.savefig('test2.png')

bins_num = 51
bins_max = 20
bins = np.linspace(0, bins_max, bins_num)
bin_width = bins_max/(bins_num-1)
bins_fine = np.linspace(0, bins_max, 501)
N_rtf1, S_rtf1 = fatpack.find_range_count(ranges_corrected_rtf1, bins)
N_rtf2, S_rtf2 = fatpack.find_range_count(ranges_corrected_rtf2, bins)
N_rtf3, S_rtf3 = fatpack.find_range_count(ranges_corrected_rtf3, bins)

DEL = (np.sum(N_rtf1*S_rtf1**m)/Neq)**(1/m)
print(DEL)
DEL = (np.sum(N_rtf2*S_rtf2**m)/Neq)**(1/m)
print(DEL)
DEL = (np.sum(N_rtf3*S_rtf3**m)/Neq)**(1/m)
print(DEL)


pdf1 = stats.exponweib.pdf(bins_fine, fit_a1,fit_c1,fit_loc1,fit_scale1)
pdf2 = stats.exponweib.pdf(bins_fine, fit_a2,fit_c2,fit_loc2,fit_scale2)
pdf3 = stats.exponweib.pdf(bins_fine, fit_a3,fit_c3,fit_loc3,fit_scale3)


plt.figure(figsize=(12, 8),dpi=100)
# plt.plot(bins_fine,y,label = 'Fitted Weibull: a='+str(round(fit_a,2))+',c='+str(round(fit_c,2)))

plt.bar(S_rtf1, N_rtf1/(np.sum(N_rtf1)*bin_width),alpha=0.5, width=bin_width,label='Baseline')
plt.bar(S_rtf2, N_rtf2/(np.sum(N_rtf2)*bin_width),alpha=0.5, width=bin_width,label='Positive 30 degree')
plt.bar(S_rtf3, N_rtf3/(np.sum(N_rtf3)*bin_width),alpha=0.5, width=bin_width,label='Negative 30 degree')
plt.ylim([0,0.4])
plt.legend()
plt.xlabel("Rainflow range ($mN \cdot m$)")
plt.ylabel("PDF")
plt.savefig('range_histgram_30.png')



plt.figure(figsize=(12, 8),dpi=100)
plt.bar(S_rtf1, np.cumsum(S_rtf1**m*N_rtf1),alpha=0.5, width=bin_width)
plt.bar(S_rtf2, np.cumsum(S_rtf2**m*N_rtf2),alpha=0.5, width=bin_width)
plt.bar(S_rtf3, np.cumsum(S_rtf3**m*N_rtf3),alpha=0.5, width=bin_width)
plt.legend()
plt.xlabel("Rainflow range ($mN \cdot m$)")
plt.ylabel("Contribution to the 10th moment of the PDF")
plt.savefig('range_histgram_mth_30.png')