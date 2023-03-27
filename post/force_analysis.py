import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


dr = 63/32
radius = np.linspace(dr/2,63-dr/2,32)

start = 10000
end = 100000

name = ['NREL-m-long_force.h5','NREL-m-long-yaw-positive_force.h5','NREL-m-long-yaw-negative_force.h5']
total_fx = np.zeros([end-start,3,32,3,3])

total_ft = np.zeros([end-start,3,32,3,3])
total_phase = np.zeros([end-start,3,3,3])
for idx,case in enumerate(name):
    print(case)
    f = h5.File('./data/'+case, 'r')
    total_fx[:,:,:,:,idx] = f['fx'][()][start:end,:,:,:]
    total_ft[:,:,:,:,idx] = f['ft'][()][start:end,:,:,:]
    total_phase[:,:,:,idx] = f['phase'][()][start:end,0,:,:]%(2*np.pi)


np.save('total_fx.npy',total_fx)
np.save('total_phase.npy',total_phase)

# def phase_average(phase,force):
#     smoothed_data = np.zeros([200,32,3])
#     for i in range(32):
#         print(i)
#         for j in range(3):
#             test = np.vstack((phase[:,0,j],force[:,0,i,j])).T
#             test1 = np.vstack((phase[:,1,j],force[:,1,i,j])).T
#             test2 = np.vstack((phase[:,2,j],force[:,2,i,j])).T
#             data = np.vstack([test,test1,test2])
#             data = data[data[:, 0].argsort()]
#             yhat = savgol_filter(data[:,1], 801, 3) 
#             xnew = np.linspace(data[0,0],data[-1,0],200)
#             f = interp1d(data[:,0],yhat, kind='nearest')
#             smoothed_data[:,i,j] = f(xnew)
#     return smoothed_data,xnew

# total_smooth_fx = np.zeros([200,32,3,3])
# total_smooth_ft = np.zeros([200,32,3,3])
# total_xnew = np.zeros([200,3])
# for i in range(3):
#     total_smooth_fx[:,:,:,i],total_xnew[:,i] = phase_average(total_phase[:,:,:,i],total_fx[:,:,:,:,i])
#     total_smooth_ft[:,:,:,i],total_xnew[:,i] = phase_average(total_phase[:,:,:,i],total_ft[:,:,:,:,i])


# # xnew = np.linspace(data[0,0],data[-1,0],200)
# # XX,YY = np.meshgrid(total_xnew[:,i],radius,indexing='ij')
# yaw_array = np.asarray([[0,0,0],[-25,-15,0],[25,15,0]])

# for k in range(3):
#     XX,YY = np.meshgrid(total_xnew[:,0],radius,indexing='ij')
#     max_value = np.max(total_smooth_fx[:,:,k,:])
#     fig = plt.figure(figsize=(18, 10), dpi=200)
#     plt.rcParams.update({'font.size': 18})
#     ax1 = fig.add_subplot(1,3,1,projection='polar')
#     im1 = ax1.contourf(XX,YY,total_smooth_fx[:,:,k,0],40,vmin=0,vmax=max_value)
#     ax1.set_title('$\gamma=$'+str(yaw_array[0,k])+'$^\circ$')

#     ax2 = fig.add_subplot(1,3,2,projection='polar')
#     im2 = ax2.contourf(XX,YY,total_smooth_fx[:,:,k,1],40,vmin=0,vmax=max_value)
#     ax2.set_title('$\gamma=$'+str(yaw_array[1,k])+'$^\circ$')

#     ax3 = fig.add_subplot(1,3,3,projection='polar')
#     im3 = ax3.contourf(XX,YY,total_smooth_fx[:,:,k,2],40,vmin=0,vmax=max_value)
#     ax3.set_title('$\gamma=$'+str(yaw_array[2,k])+'$^\circ$')

#     cbar_ax = fig.add_axes([0.12, 0.15, 0.8, 0.05])
#     clb = fig.colorbar(im2, cax=cbar_ax,orientation='horizontal')
#     clb.ax.set_title('Normal force ($kN\cdot m$)',fontsize=18)
#     plt.savefig('./plot/force-moment-'+str(k)+'.png')
# # print(smooth_fx[:,:,0].shape)
