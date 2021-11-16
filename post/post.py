import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import fatpack
import fatigue

root_moment_flap0 = pd.read_csv('../job/NREL-m/src/output/root_moment.csv',header=None).iloc[0:-1].astype(float).to_numpy()
root_moment_flap1 = pd.read_csv('../job/NREL-m-yaw1/src/output/root_moment.csv',header=None).iloc[0:-1].astype(float).to_numpy()
root_moment_flap2 = pd.read_csv('../job/NREL-m-yaw2/src/output/root_moment.csv',header=None).iloc[0:-1].astype(float).to_numpy()

root_moment_flap0 = root_moment_flap0[2000:3000]
root_moment_flap1 = root_moment_flap1[2000:3000]
root_moment_flap2 = root_moment_flap2[2000:3000]

# root_force_flap  = pd.read_csv('../job/NREL-m/src/output/root.csv',header=None).iloc[0:-1].astype(float).to_numpy()
# root_force_flap2 = pd.read_csv('../job/NREL-m-yaw1/src/output/root.csv',header=None).iloc[0:-1].astype(float).to_numpy()
# root_force_flap0 = pd.read_csv('../job/NREL-m-yaw2/src/output/root.csv',header=None).iloc[0:-1].astype(float).to_numpy()

# print(np.mean(root_force_flap[10000:]))
# print(np.mean(root_force_flap2[10000:]))
# print(np.mean(root_force_flap0[10000:]))

fig,ax = plt.subplots(1,1)
plt.plot(root_moment_flap0,lw=1)
plt.plot(root_moment_flap1,lw=1)
plt.plot(root_moment_flap2,lw=1)
# plt.xlim([8000,10000])
# plt.xlim([1000,3000])
# plt.ylim([4e6,7e6])
# fig,ax = plt.subplots(1,1)
# plt.rcParams['image.cmap']='Purples'
# value_plot = u[0,128,:,:].T
# plt.contourf(value_plot,100)
# ax.set_aspect(0.5)

# plt.clim(np.amin(value_plot),np.amax(value_plot)*1.6)
plt.savefig('moment.png')


# fig,ax = plt.subplots(1,1)
# plt.plot(root_force_flap)
# plt.plot(root_force_flap2)
# plt.plot(root_force_flap0)
# # plt.xlim([8000,10000])
# # plt.ylim([4e6,7e6])
# # fig,ax = plt.subplots(1,1)
# # plt.rcParams['image.cmap']='Purples'
# # value_plot = u[0,128,:,:].T
# # plt.contourf(value_plot,100)
# # ax.set_aspect(0.5)

# # plt.clim(np.amin(value_plot),np.amax(value_plot)*1.6)
# plt.savefig('force.png')
Neq = 100
M_eq_baseline = fatigue.get_DEL(root_moment_flap0,Neq,10)
print(M_eq_baseline)
M_eq_baseline = fatigue.get_DEL(root_moment_flap1,Neq,10)
print(M_eq_baseline)
M_eq_baseline = fatigue.get_DEL(root_moment_flap2,Neq,10)
print(M_eq_baseline)