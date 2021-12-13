import numpy as np
import math
import fatpack
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import seaborn as sns
from scipy.signal import savgol_filter

def Goodman_method_correction(M_a,M_m,M_max):
    M_u = 1.5*M_max
    M_ar = M_a/(1-M_m/M_u)
    return M_ar

def Equivalent_bending_moment(M_ar,Neq,m):
    P = M_ar.shape
    M_sum = 0
    j = P[0] 
    for i in range(j):
        M_sum = math.pow(M_ar[i],m) + M_sum
    M_eq = math.pow((M_sum/Neq),(1/m))
    return M_eq

def get_DEL(y,Neq,m):
    S, Sm = fatpack.find_rainflow_ranges(y.flatten(), return_means=True, k=256)
    M_ar = Goodman_method_correction(S,Sm,np.max(y))
    hist, bin_edges = np.histogram(M_ar,bins=100)
    bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
    DEL = (np.sum(hist*bin_centres**m)/Neq)**(1/m)
    return DEL

f0 = h5py.File('../job/3wt-strong-shear/output/3wt-strong-shear_force.h5','r')
f1 = h5py.File('../job/3wt+1/output/3wt+1_force.h5','r')
f2 = h5py.File('../job/3wt+2/output/3wt+2_force.h5','r')
f3 = h5py.File('../job/3wt+3/output/3wt+3_force.h5','r')
f4 = h5py.File('../job/3wt+4/output/3wt+4_force.h5','r')
f5 = h5py.File('../job/3wt-strong-shear-yaw1/output/3wt-strong-shear-yaw1_force.h5','r')
f6 = h5py.File('../job/3wt+6/output/3wt+6_force.h5','r')
f7 = h5py.File('../job/3wt+7/output/3wt+7_force.h5','r')
f8 = h5py.File('../job/3wt+8/output/3wt+8_force.h5','r')
f9 = h5py.File('../job/3wt+9/output/3wt+9_force.h5','r')
f10 = h5py.File('../job/3wt-strong-shear-yaw2/output/3wt-strong-shear-yaw2_force.h5','r')
f11 = h5py.File('../job/3wt+11/output/3wt+11_force.h5','r')
f12 = h5py.File('../job/3wt+12/output/3wt+12_force.h5','r')
f13 = h5py.File('../job/3wt+13/output/3wt+13_force.h5','r')
f14 = h5py.File('../job/3wt+14/output/3wt+14_force.h5','r')
f15 = h5py.File('../job/3wt-strong-shear-yaw3/output/3wt-strong-shear-yaw3_force.h5','r')
f16 = h5py.File('../job/3wt+16/output/3wt+16_force.h5','r')
f17 = h5py.File('../job/3wt+17/output/3wt+17_force.h5','r')
f18 = h5py.File('../job/3wt+18/output/3wt+18_force.h5','r')
f19 = h5py.File('../job/3wt+19/output/3wt+19_force.h5','r')
f20 = h5py.File('../job/3wt-strong-shear-yaw4/output/3wt-strong-shear-yaw4_force.h5','r')


m_f0 = np.array(f0.get('moment_flap'))
m_f1 = np.array(f1.get('moment_flap'))
m_f2 = np.array(f2.get('moment_flap'))
m_f3 = np.array(f3.get('moment_flap'))
m_f4 = np.array(f4.get('moment_flap'))
m_f5 = np.array(f5.get('moment_flap'))
m_f6 = np.array(f6.get('moment_flap'))
m_f7 = np.array(f7.get('moment_flap'))
m_f8 = np.array(f8.get('moment_flap'))
m_f9 = np.array(f9.get('moment_flap'))
m_f10 = np.array(f10.get('moment_flap'))
m_f11 = np.array(f11.get('moment_flap'))
m_f12 = np.array(f12.get('moment_flap'))
m_f13 = np.array(f13.get('moment_flap'))
m_f14 = np.array(f14.get('moment_flap'))
m_f15 = np.array(f15.get('moment_flap'))
m_f16 = np.array(f16.get('moment_flap'))
m_f17 = np.array(f17.get('moment_flap'))
m_f18 = np.array(f18.get('moment_flap'))
m_f19 = np.array(f19.get('moment_flap'))
m_f20 = np.array(f20.get('moment_flap'))

m = 10
Neq = 1000
start = 20000

# y1 = m_f1[start:,0,0,0]
# y2 = m_f2[start:,0,0,0]
# y3 = m_f3[start:,0,0,0]
DEL = np.zeros([21,3])
for i in range(3):
    print(i)
    DEL[0,i] = get_DEL(m_f0[start:,0,0,i],Neq,m)
    DEL[1,i] = get_DEL(m_f1[start:,0,0,i],Neq,m)
    DEL[2,i] = get_DEL(m_f2[start:,0,0,i],Neq,m)
    DEL[3,i] = get_DEL(m_f3[start:,0,0,i],Neq,m)
    DEL[4,i] = get_DEL(m_f4[start:,0,0,i],Neq,m)
    DEL[5,i] = get_DEL(m_f5[start:,0,0,i],Neq,m)
    DEL[6,i] = get_DEL(m_f6[start:,0,0,i],Neq,m)
    DEL[7,i] = get_DEL(m_f7[start:,0,0,i],Neq,m)
    DEL[8,i] = get_DEL(m_f8[start:,0,0,i],Neq,m)
    DEL[9,i] = get_DEL(m_f9[start:,0,0,i],Neq,m)
    DEL[10,i] = get_DEL(m_f10[start:,0,0,i],Neq,m)
    DEL[11,i] = get_DEL(m_f11[start:,0,0,i],Neq,m)
    DEL[12,i] = get_DEL(m_f12[start:,0,0,i],Neq,m)
    DEL[13,i] = get_DEL(m_f13[start:,0,0,i],Neq,m)
    DEL[14,i] = get_DEL(m_f14[start:,0,0,i],Neq,m)
    DEL[15,i] = get_DEL(m_f15[start:,0,0,i],Neq,m)
    DEL[16,i] = get_DEL(m_f16[start:,0,0,i],Neq,m)
    DEL[17,i] = get_DEL(m_f17[start:,0,0,i],Neq,m)
    DEL[18,i] = get_DEL(m_f18[start:,0,0,i],Neq,m)
    DEL[19,i] = get_DEL(m_f19[start:,0,0,i],Neq,m)
    DEL[20,i] = get_DEL(m_f20[start:,0,0,i],Neq,m)



f0 = h5py.File('../job/3wt-strong-shear/output/3wt-strong-shear_force.h5','r')
f1 = h5py.File('../job/3wt-1/output/3wt-1_force.h5','r')
f2 = h5py.File('../job/3wt-2/output/3wt-2_force.h5','r')
f3 = h5py.File('../job/3wt-3/output/3wt-3_force.h5','r')
f4 = h5py.File('../job/3wt-4/output/3wt-4_force.h5','r')
f5 = h5py.File('../job/3wt-strong-shear-yaw5/output/3wt-strong-shear-yaw5_force.h5','r')
f6 = h5py.File('../job/3wt-6/output/3wt-6_force.h5','r')
f7 = h5py.File('../job/3wt-7/output/3wt-7_force.h5','r')
f8 = h5py.File('../job/3wt-8/output/3wt-8_force.h5','r')
f9 = h5py.File('../job/3wt-9/output/3wt-9_force.h5','r')
f10 = h5py.File('../job/3wt-strong-shear-yaw6/output/3wt-strong-shear-yaw6_force.h5','r')
f11 = h5py.File('../job/3wt-11/output/3wt-11_force.h5','r')
f12 = h5py.File('../job/3wt-12/output/3wt-12_force.h5','r')
f13 = h5py.File('../job/3wt-13/output/3wt-13_force.h5','r')
f14 = h5py.File('../job/3wt-14/output/3wt-14_force.h5','r')
f15 = h5py.File('../job/3wt-strong-shear-yaw7/output/3wt-strong-shear-yaw7_force.h5','r')
f16 = h5py.File('../job/3wt-16/output/3wt-16_force.h5','r')
f17 = h5py.File('../job/3wt-17/output/3wt-17_force.h5','r')
f18 = h5py.File('../job/3wt-18/output/3wt-18_force.h5','r')
f19 = h5py.File('../job/3wt-19/output/3wt-19_force.h5','r')
f20 = h5py.File('../job/3wt-strong-shear-yaw8/output/3wt-strong-shear-yaw8_force.h5','r')


m_f0 = np.array(f0.get('moment_flap'))
m_f1 = np.array(f1.get('moment_flap'))
m_f2 = np.array(f2.get('moment_flap'))
m_f3 = np.array(f3.get('moment_flap'))
m_f4 = np.array(f4.get('moment_flap'))
m_f5 = np.array(f5.get('moment_flap'))
m_f6 = np.array(f6.get('moment_flap'))
m_f7 = np.array(f7.get('moment_flap'))
m_f8 = np.array(f8.get('moment_flap'))
m_f9 = np.array(f9.get('moment_flap'))
m_f10 = np.array(f10.get('moment_flap'))
m_f11 = np.array(f11.get('moment_flap'))
m_f12 = np.array(f12.get('moment_flap'))
m_f13 = np.array(f13.get('moment_flap'))
m_f14 = np.array(f14.get('moment_flap'))
m_f15 = np.array(f15.get('moment_flap'))
m_f16 = np.array(f16.get('moment_flap'))
m_f17 = np.array(f17.get('moment_flap'))
m_f18 = np.array(f18.get('moment_flap'))
m_f19 = np.array(f19.get('moment_flap'))
m_f20 = np.array(f20.get('moment_flap'))

m = 10
Neq = 1000
start = 20000

# y1 = m_f1[start:,0,0,0]
# y2 = m_f2[start:,0,0,0]
# y3 = m_f3[start:,0,0,0]
DEL2 = np.zeros([21,3])
for i in range(3):
    print(i)
    DEL2[0,i] = get_DEL(m_f0[start:,0,0,i],Neq,m)
    DEL2[1,i] = get_DEL(m_f1[start:,0,0,i],Neq,m)
    DEL2[2,i] = get_DEL(m_f2[start:,0,0,i],Neq,m)
    DEL2[3,i] = get_DEL(m_f3[start:,0,0,i],Neq,m)
    DEL2[4,i] = get_DEL(m_f4[start:,0,0,i],Neq,m)
    DEL2[5,i] = get_DEL(m_f5[start:,0,0,i],Neq,m)
    DEL2[6,i] = get_DEL(m_f6[start:,0,0,i],Neq,m)
    DEL2[7,i] = get_DEL(m_f7[start:,0,0,i],Neq,m)
    DEL2[8,i] = get_DEL(m_f8[start:,0,0,i],Neq,m)
    DEL2[9,i] = get_DEL(m_f9[start:,0,0,i],Neq,m)
    DEL2[10,i] = get_DEL(m_f10[start:,0,0,i],Neq,m)
    DEL2[11,i] = get_DEL(m_f11[start:,0,0,i],Neq,m)
    DEL2[12,i] = get_DEL(m_f12[start:,0,0,i],Neq,m)
    DEL2[13,i] = get_DEL(m_f13[start:,0,0,i],Neq,m)
    DEL2[14,i] = get_DEL(m_f14[start:,0,0,i],Neq,m)
    DEL2[15,i] = get_DEL(m_f15[start:,0,0,i],Neq,m)
    DEL2[16,i] = get_DEL(m_f16[start:,0,0,i],Neq,m)
    DEL2[17,i] = get_DEL(m_f17[start:,0,0,i],Neq,m)
    DEL2[18,i] = get_DEL(m_f18[start:,0,0,i],Neq,m)
    DEL2[19,i] = get_DEL(m_f19[start:,0,0,i],Neq,m)
    DEL2[20,i] = get_DEL(m_f20[start:,0,0,i],Neq,m)





# # print(DEL1,DEL2,DEL3)
plt.figure()
plt.plot(DEL[:,0])
plt.plot(DEL2[:,0])
# plt.plot(m_f1[start:,0,0,0])
# plt.plot(m_f2[start:,0,0,0])
# plt.plot(m_f3[start:,0,0,0])
plt.savefig('test.png')