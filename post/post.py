import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import fatpack
import fatigue

root_moment0 = pd.read_csv('../job/NREL-m/output/root_moment.csv',header=None).iloc[0:-1].astype(float).to_numpy()
root_moment1 = pd.read_csv('../job/NREL-m-yaw1/output/root_moment.csv',header=None).iloc[0:-1].astype(float).to_numpy()
root_moment2 = pd.read_csv('../job/NREL-m-yaw2/output/root_moment.csv',header=None).iloc[0:-1].astype(float).to_numpy()

root_moment0 = root_moment0[1000:10000] 
root_moment1 = root_moment1[1000:10000] 
root_moment2 = root_moment2[1000:10000] 

fig,ax = plt.subplots(1,1)
plt.plot(root_moment0[:,0],lw=1)
plt.plot(root_moment1[:,0],lw=1)
plt.plot(root_moment2[:,0],lw=1)
# plt.xlim([8000,10000])
plt.savefig('flap_moment.png')


fig,ax = plt.subplots(1,1)
plt.plot(root_moment0[:,1],lw=1)
plt.plot(root_moment1[:,1],lw=1)
plt.plot(root_moment2[:,1],lw=1)
plt.xlim([8000,10000])
plt.savefig('edge_moment.png')


# fig,ax = plt.subplots(1,1)
# plt.plot(root_force_flap)
# plt.plot(root_force_flap2)
# plt.plot(root_force_flap0)
plt.xlim([2000,10000])
# # plt.ylim([4e6,7e6])
# # fig,ax = plt.subplots(1,1)
# # plt.rcParams['image.cmap']='Purples'
# # value_plot = u[0,128,:,:].T
# # plt.contourf(value_plot,100)
# # ax.set_aspect(0.5)

# # plt.clim(np.amin(value_plot),np.amax(value_plot)*1.6)
# plt.savefig('force.png')
Neq = 1000
M_eq_baseline = fatigue.get_DEL(root_moment0[:,0],Neq,10)
print(M_eq_baseline)
M_eq_baseline = fatigue.get_DEL(root_moment1[:,0],Neq,10)
print(M_eq_baseline)
M_eq_baseline = fatigue.get_DEL(root_moment2[:,0],Neq,10)
print(M_eq_baseline)
# M_eq_baseline = fatigue.get_DEL(root_moment2[:,0],Neq,10)
# print(M_eq_baseline)