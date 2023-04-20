import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

f = h5.File('./job/HR1-m/output/HR1-m_stat.h5','r')
u_avg = f['u_avg'][:]
u_std = f['u_std'][:]

inflow = np.mean(u_avg[30:50,:,6])
inflow_ti = np.mean(u_std[30:50,:,6])
print(inflow)
print(inflow_ti/inflow)
plt.figure(dpi=300)
plt.imshow((u_avg[:,:,6].T),origin='lower',extent=[0,10.24,0,5.12],aspect=1/1)
plt.xlabel('x (km)')
plt.ylabel('y (km)')
plt.xlim([2,8])
plt.ylim([0,5.12])
plt.colorbar()
plt.show()