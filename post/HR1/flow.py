import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

f = h5.File('../../job/HR1-m-fine/output/HR1-m-fine_stat.h5', 'r')

# print(list(f.keys()))
u_avg = f['u_avg'][:]
# print(u_avg.shape)
plt.imshow(np.flip(u_avg[:,:,7].T,axis=0),origin='lower',extent=[0,10240,0,5120],vmin=0,vmax=8)
plt.savefig('test.png')