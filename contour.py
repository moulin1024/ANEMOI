import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys

yaw_angle = sys.argv[1]
f = h5.File('./job/HR1-m-'+yaw_angle+'/output/HR1-m-'+yaw_angle+'_stat.h5','r')
u_avg = f['u_avg'][:]
u_std = f['u_std'][:]

file_path = './job/HR1-m-'+yaw_angle+'/src/output/ta_power.dat'
power = np.loadtxt(file_path)
total_power = np.sum(power)/1e6
print(total_power)


inflow = np.mean(u_avg[30:50,:,6])
inflow_ti = np.mean(u_std[30:50,:,6])
print(inflow)
print(inflow_ti/inflow)
plt.figure(dpi=300)
plt.imshow((u_avg[:,:,6].T),origin='lower',extent=[0,10.24,0,5.12],vmin=2,vmax=8,aspect=1/1)
plt.xlabel('x (km)')
plt.title('Power: '+str(round(total_power,2))+' MW')
plt.ylabel('y (km)')
plt.xlim([1,9])
plt.ylim([0,5.12])
plt.colorbar()
plt.savefig('contour_'+yaw_angle+'.png')