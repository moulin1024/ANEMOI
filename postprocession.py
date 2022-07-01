import numpy as np
import matplotlib.pyplot as plt
import h5py

f = h5py.File('dyn-yaw-8wt-baseline_stat.h5','r')
u1 = np.array(f.get('u_avg')[:,:,44])

f = h5py.File('dyn-yaw-8wt-30-360s_stat.h5','r')
u2 = np.array(f.get('u_avg')[:,:,44])
x = np.array(f.get('x'))
y = np.array(f.get('y'))

fig,ax = plt.subplots(2,1,figsize=(8,4),dpi=300)
# for i in range()
ax[0].contourf(x,y,u1.T,100,vmin=4,vmax=12)
ax[1].contourf(x,y,u2.T,100,vmin=4,vmax=12)
# plt.colorbar()
ax[0].axis('scaled')
ax[1].axis('scaled')
ax[0].set_xlabel('x (m)')
ax[0].set_title('Baseline')
ax[1].set_title('Cyclic yaw')
ax[0].set_ylabel('y (m)')
ax[1].set_xlabel('x (m)')
ax[1].set_ylabel('y (m)')
plt.savefig('test.png')

fig = plt.figure(figsize=(14,6),dpi=300)
plt.rcParams.update({'font.size': 18})
plt.plot(x[20:],u1[20:,64],label='Baseline')
plt.plot(x[20:],u2[20:,64],label='Cyclic yaw')
for i in range(8):
    plt.plot([1024+i*128*7,1024+i*128*7],[0,12],'k--',lw=1)
    plt.text(1024+10+i*128*7,11,'WT'+str(i+1))
plt.ylim(4,12)
plt.xlabel('x (m)')
plt.ylabel('u (m/s)')
plt.legend(loc=(0.25,1.02),ncol=2)
# plt.rcParams.update({'font.size': 40})
plt.savefig('test1.png')