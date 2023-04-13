import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

f = h5.File('./job/HR1/output/HR1_stat.h5','r')
u_avg = f['u_avg'][:]
u_std = f['u_std'][:]
height = np.asarray(np.arange(127)+0.5)*(1280/127)
u_mean = np.mean(np.mean(u_avg,axis=0),axis=0)
uu_mean = np.mean(np.mean(u_std,axis=0),axis=0)
u_log_law = 0.29/0.4*np.log(height/0.001)
plt.semilogy(u_mean,height,'o-')
plt.semilogy(u_log_law,height,'--')

plt.savefig('test.png')

from scipy.interpolate import interp1d

# plt.plot(uu_mean/8,height,'.')
# plt.plot()

interp_func = interp1d(height, u_mean)
u_hub = interp_func(70)
print(u_hub)

interp_func = interp1d(height, uu_mean)
uu_hub = interp_func(70)
print(uu_hub/u_hub)