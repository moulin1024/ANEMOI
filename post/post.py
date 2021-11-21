import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import fatpack
import h5py
import fatigue


f = h5py.File('data/NREL-m-test.h5','r')

u = np.array(f.get('velocity/u'))
v = np.array(f.get('velocity/v'))
w = np.array(f.get('velocity/w'))

x = np.array(f.get('space/x'))
y = np.array(f.get('space/y'))
z = np.array(f.get('space/z')) 

x_mask = x[f.attrs['ts_istart']-1:f.attrs['ts_iend']]
y_mask = y[f.attrs['ts_jstart']-1:f.attrs['ts_jend']]
z_mask = z[:f.attrs['ts_kend']-1]

fig = figure(figsize=(11,6),dpi=200)
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(221)
ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224,projection='3d')
# fig,ax = plt.subplots(1,1)
def animate(i):    
    plt.cla()
    ax1.contourf(y_mask,z_mask,u[i,64,:,:].T,50)
    ax1.set_xlabel('y')
    ax1.set_ylabel('z')

    ax2.contourf(x_mask,z_mask,v[i,:,16,:].T,50)
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')

    ax3.contourf(x_mask,y_mask,w[i,:,:,45].T,50)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    ax1.axis('scaled')
    ax2.axis('scaled')
    ax3.axis('scaled')
    print(i)
    plt.tight_layout()
    return
anim = animation.FuncAnimation(fig, animate, frames=10)
anim.save('plot/animation_xz.gif',writer='pillow', fps=10)