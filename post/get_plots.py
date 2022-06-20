import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.pyplot import figure
from matplotlib import animation, rc
import fatpack

power = np.load('power.npy')
total_power = np.sum(power,axis=3)
print(total_power.shape)

#power_map
climt = 0.1
plt.figure(figsize=(10, 8), dpi=300)
levels = np.linspace(-climt,climt,101)
ticklevel = np.linspace(-climt,climt,6)
plt.rcParams.update({'font.size': 22})
total_power_ref = total_power[6,6,5]
normal_power_ref = total_power[:,:,5]/total_power_ref-1
normal_power_ref = np.flip(np.flip(normal_power_ref,axis=0),axis=1)
max_power = np.max(np.max(np.max(normal_power_ref)))
max_power_2 = np.max(np.max(np.max(normal_power_ref[6:,6:])))
i,j = np.where(normal_power_ref == max_power)
i2,j2 = np.where(normal_power_ref == max_power_2)
max_name = '$('+str((i[0]-6)*5)+'^\circ'+','+str((j[0]-6)*5)+'^\circ'+')$: '+str(round(max_power*100,2))+'%'
max_name_2 = '$('+str((i2[0]-6)*5)+'^\circ'+','+str((j2[0]-6)*5)+'^\circ'+')$: '+str(round(max_power_2*100,2))+'%'
plt.contourf(normal_power_ref.T,levels,extent=[-30,30,-30,30],vmax=climt,vmin=-climt,cmap='bwr',extend='both')
plt.plot((i2[0]-6)*5,(j2[0]-6)*5,'ro',label=max_name_2,markersize=10)
plt.plot((i[0]-6)*5,(j[0]-6)*5,'bo',label=max_name,markersize=10)
plt.colorbar(ticks=ticklevel, format='%.2f')
plt.clim(-climt,climt)
plt.title('$\Delta P/P_{0}$',fontsize=28)
plt.xlabel('$\gamma_1$ ($ ^\circ$)')
plt.ylabel('$\gamma_2$ ($ ^\circ$)')
plt.legend(fontsize=24,loc='upper left')
plt.savefig('power_ref.png',bbox='tight')


#DEL map
DEL = np.load('DEL.npy')
total_DEL = np.mean(np.sum(DEL,axis=4),axis=3)
# total_DEL = np.mean(DEL[:,:,:,:,0],axis=3)
print(total_DEL.shape)
DEL_ref = total_DEL[6,6,5]
normal_DEL_ref = -np.flip(np.flip(1-total_DEL[:,:,5]/DEL_ref,axis=0),axis=1)
print(normal_DEL_ref.shape)
climt = 0.3
plt.figure(figsize=(10, 8), dpi=300)
levels = np.linspace(-climt,climt,101)
ticklevel = np.linspace(-climt,climt,6)
plt.rcParams.update({'font.size': 22})
plt.contourf(normal_DEL_ref.T,levels,extent=[-30,30,-30,30],vmax=climt,vmin=-climt,cmap='bwr',extend='both')
plt.colorbar(ticks=ticklevel, format='%.2f')
plt.clim(-climt,climt)
max_name = '$('+str((i[0]-6)*5)+'^\circ'+','+str((j[0]-6)*5)+'^\circ'+')$: '+str(round(normal_DEL_ref[i[0],j[0]]*100,2))+'%'
max_name_2 = '$('+str((i2[0]-6)*5)+'^\circ'+','+str((j2[0]-6)*5)+'^\circ'+')$: '+str(round(normal_DEL_ref[i2[0],j2[0]]*100,2))+'%'
plt.plot((i2[0]-6)*5,(j2[0]-6)*5,'ro',label=max_name_2,markersize=10)
plt.plot((i[0]-6)*5,(j[0]-6)*5,'bo',label=max_name,markersize=10)
plt.title('$\Delta DEL/DEL_{0}$',fontsize=28)
plt.xlabel('$\gamma_1$ ($ ^\circ$)')
plt.ylabel('$\gamma_2$ ($ ^\circ$)')
plt.legend(fontsize=24,loc='upper left')
plt.savefig('DEL_ref.png')



# power_ref = power[6,6,5,:]/power[6,6,5,0]
# power_opti_positive = power[i[0],j[0],5,:]/power[6,6,5,0]
# power_opti_negative = power[i2[0],j2[0],5,:]/power[6,6,5,0]
# wt_name=['wt1','wt2','wt3']
# x = np.asarray([0,1,2])
# width = 0.3 
# print(power_ref)
# print(power_opti_positive)
# plt.figure()
# plt.bar(x-width,power_ref,width,label='$0^\circ,0^\circ,0^\circ$')
# plt.bar(x,power_opti_positive,width,label='$25^\circ,15^\circ,0^\circ$')
# plt.bar(x+width,power_opti_negative,width,label='$-25^\circ,-15^\circ,0^\circ$')
# plt.xticks(x,wt_name)
# plt.ylabel('Normalised power')
# plt.legend()
# plt.savefig('power_bar_ref.png')


# mean_DEL = np.mean(DEL,axis=3)
# print(mean_DEL.shape)
# DEL_ref = mean_DEL[6,6,5,:]/mean_DEL[6,6,5,0]
# DEL_opti_positive = mean_DEL[i[0],j[0],5,:]/mean_DEL[6,6,5,0]
# DEL_opti_negative = mean_DEL[i2[0],j2[0],5,:]/mean_DEL[6,6,5,0]
# wt_name=['wt1','wt2','wt3']
# x = np.asarray([0,1,2])
# width = 0.3 
# print(DEL_ref)
# print(DEL_opti_positive)
# plt.figure()
# plt.bar(x-width,DEL_ref,width,label='$0^\circ,0^\circ,0^\circ$')
# plt.bar(x,DEL_opti_positive,width,label='$25^\circ,15^\circ,0^\circ$')
# plt.bar(x+width,DEL_opti_negative,width,label='$-25^\circ,-15^\circ,0^\circ$')
# plt.xticks(x,wt_name)
# plt.ylabel('Normalised DEL')
# plt.ylim([0,1.5])
# plt.legend()
# plt.savefig('DEL_bar_ref.png')


# name = ["rotate-0-0-0","rotate-0-25-15","rotate-0--25--15"]
# title = ["$0^\circ,0^\circ,0^\circ$","$-25^\circ,-15^\circ,0^\circ$","$25^\circ,15^\circ,0^\circ$"]

# levels = np.linspace(0.4,1.0,101)
# ticklevel = np.linspace(0.4,1.0,7)
# fig, ax = plt.subplots(1,3,figsize=(20, 4), dpi=300)
# for i in range(3):
#     f = h5py.File('../job/'+name[i]+'/output/'+name[i]+'_stat.h5','r')
#     inflow = np.mean(np.mean(f.get('u_avg')[50:100,:,44]))
#     print(inflow)

#     u = np.flip(np.array(f.get('u_avg')[:,:,44])/inflow,axis=1)
#     Iu = np.array(f.get('u_std')[:,:,44])
#     im = ax[i].contourf(u.T,levels,extent=[0,4.096,0,2.048],vmin=0.4,vmax=1,extend='both')
#     ax[i].axis('scaled')
#     ax[i].set_xlabel('x (km)')
#     ax[i].set_ylabel('y (km)')
#     ax[i].set_ylabel('y (km)')
#     ax[i].set_title(title[i])
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85,0.15, 0.02, 0.7])
#     fig.colorbar(im, cax=cbar_ax,ticks=ticklevel, format='%.2f')

# plt.savefig('mean_u_ref.png')



# levels = np.linspace(0.06,0.2,101)
# ticklevel = np.linspace(0.06,0.2,8)
# fig, ax = plt.subplots(1,3,figsize=(20, 4), dpi=300)
# for i in range(3):
#     f = h5py.File('../job/'+name[i]+'/output/'+name[i]+'_stat.h5','r')
#     inflow = np.mean(np.mean(f.get('u_avg')[50:100,:,44]))
#     print(inflow)

#     Iu = np.flip(np.array(f.get('u_std')[:,:,44]),axis=1)/inflow
#     im = ax[i].contourf(Iu.T,levels,extent=[0,4.096,0,2.048],vmin=0.06,vmax=0.2,extend='both')
#     ax[i].axis('scaled')
#     ax[i].set_xlabel('x (km)')
#     ax[i].set_ylabel('y (km)')
#     ax[i].set_ylabel('y (km)')
#     ax[i].set_title(title[i])
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85,0.15, 0.02, 0.7])
#     fig.colorbar(im, cax=cbar_ax,ticks=ticklevel, format='%.2f')
# plt.savefig('mean_uu_ref.png')



# power = np.load('power.npy')
# total_power = np.sum(power,axis=3)
# print(total_power.shape)

# #power_map
# climt = 0.15
# fig, ax = plt.subplots(2,5,figsize=(25, 10), dpi=300)
# # plt.rcParams.update({'font.size': 18})
# levels = np.linspace(-climt,climt,101)
# ticklevel = np.linspace(-climt,climt,6)
# # plt.rcParams.update({'font.size': 22})
# case_idx_array = np.asarray([[4,3,2,1,0],[6,7,8,9,10]])
# location = ['lower left','upper left']
# for i in range(5):
#     for j in range(2):
#         case_idx = case_idx_array[j,i]
#         print(case_idx)
#         total_power_ref = total_power[6,6,case_idx]
#         normal_power_ref = total_power[:,:,case_idx]/total_power_ref-1
#         normal_power_ref = np.flip(np.flip(normal_power_ref,axis=0),axis=1)
#         max_power = np.max(np.max(np.max(normal_power_ref)))
#         # max_power_2 = np.max(np.max(np.max(normal_power_ref[6:,6:])))
#         ii,jj = np.where(normal_power_ref == max_power)
#         # i2,j2 = np.where(normal_power_ref == max_power_2)
#         max_name = '$('+str((ii[0]-6)*5)+'^\circ'+','+str((jj[0]-6)*5)+'^\circ'+')$: \n $\Delta P/P_{0}$= '+str(round(max_power*100,2))+'%'
#         # max_name_2 = '$('+str((i2[0]-6)*5)+'^\circ'+','+str((j2[0]-6)*5)+'^\circ'+')$: $\Delta P/P_{0}$= '+str(round(max_power_2*100,2))+'%'
#         # print(i)
#         im = ax[j,i].contourf(normal_power_ref.T,levels,extent=[-30,30,-30,30],vmax=climt,vmin=-climt,cmap='bwr',extend='both')
#         # plt.plot((i2[0]-6)*5,(j2[0]-6)*5,'o',label=max_name_2)
#         ax[j,i].plot((ii[0]-6)*5,(jj[0]-6)*5,'o',label=max_name)
# # plt.colorbar(ticks=ticklevel, format='%.2f')
# # plt.clim(-climt,climt)
#         plt.title('inflow angle: '+str()+'$^\circ$')
#         if (j == 1):
#             ax[j,i].set_xlabel('$\gamma_1$ ($ ^\circ$)')
#         if (i == 0):
#             ax[j,i].set_ylabel('$\gamma_2$ ($ ^\circ$)')
        
#         if (j == 0):
#             ax[j,i].set_title('inflow angle: '+str(i+1)+'$^\circ$')
#         if (j == 1):
#             ax[j,i].set_title('inflow angle: '+str(-i-1)+'$^\circ$')
#         ax[j,i].legend(fontsize=16,loc=location[j])

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85,0.15, 0.02, 0.7])
# fig.colorbar(im, cax=cbar_ax,ticks=ticklevel, format='%.2f')
# plt.savefig('power_rotate_ref.png')



# DEL = np.load('DEL.npy')
# mean_DEL = np.mean(DEL,axis=3)
# total_DEL = np.sum(mean_DEL,axis=3)
# print(total_DEL.shape)

# #power_map
# climt = 0.3
# fig, ax = plt.subplots(2,5,figsize=(25, 10), dpi=300)
# # plt.rcParams.update({'font.size': 18})
# levels = np.linspace(-climt,climt,101)
# ticklevel = np.linspace(-climt,climt,6)
# # plt.rcParams.update({'font.size': 22})
# case_idx_array = np.asarray([[4,3,2,1,0],[6,7,8,9,10]])
# location = ['lower left','upper left']
# for i in range(5):
#     for j in range(2):
#         case_idx = case_idx_array[j,i]
#         print(case_idx)
#         total_DEL_ref = total_DEL[6,6,case_idx]
#         normal_DEL_ref = 1-total_DEL[:,:,case_idx]/total_DEL_ref
#         normal_DEL_ref = np.flip(np.flip(normal_DEL_ref,axis=0),axis=1)

#         total_power_ref = total_power[6,6,case_idx]
#         normal_power_ref = total_power[:,:,case_idx]/total_power_ref-1
#         normal_power_ref = np.flip(np.flip(normal_power_ref,axis=0),axis=1)
#         max_power = np.max(np.max(np.max(normal_power_ref)))
#         # max_power_2 = np.max(np.max(np.max(normal_power_ref[6:,6:])))
#         ii,jj = np.where(normal_power_ref == max_power)
#         # i2,j2 = np.where(normal_power_ref == max_power_2)
#         max_name = '$('+str((ii[0]-6)*5)+'^\circ'+','+str((jj[0]-6)*5)+'^\circ'+')$: \n $-\Delta DEL/DEL_{0}$= '+str(round(normal_DEL_ref[ii[0],jj[0]]*100,2))+'%'
        
#         # max_power = np.max(np.max(np.max(normal_power_ref)))
#         # max_power_2 = np.max(np.max(np.max(normal_power_ref[6:,6:])))
#         # ii,jj = np.where(normal_power_ref == max_power)
#         # i2,j2 = np.where(normal_power_ref == max_power_2)
#         # max_name = '$('+str((ii[0]-6)*5)+'^\circ'+','+str((jj[0]-6)*5)+'^\circ'+')$: \n $\Delta P/P_{0}$= '+str(round(max_power*100,2))+'%'
#         # max_name_2 = '$('+str((i2[0]-6)*5)+'^\circ'+','+str((j2[0]-6)*5)+'^\circ'+')$: $\Delta P/P_{0}$= '+str(round(max_power_2*100,2))+'%'
#         # print(i)
#         im = ax[j,i].contourf(normal_DEL_ref.T,levels,extent=[-30,30,-30,30],vmax=climt,vmin=-climt,cmap='bwr',extend='both')
#         # plt.plot((i2[0]-6)*5,(j2[0]-6)*5,'o',label=max_name_2)
#         ax[j,i].plot((ii[0]-6)*5,(jj[0]-6)*5,'o',label=max_name)
# # plt.colorbar(ticks=ticklevel, format='%.2f')
# # plt.clim(-climt,climt)
#         plt.title('inflow angle: '+str()+'$^\circ$')
#         if (j == 1):
#             ax[j,i].set_xlabel('$\gamma_1$ ($ ^\circ$)')
#         if (i == 0):
#             ax[j,i].set_ylabel('$\gamma_2$ ($ ^\circ$)')
        
#         if (j == 0):
#             ax[j,i].set_title('inflow angle: '+str(i+1)+'$^\circ$')
#         if (j == 1):
#             ax[j,i].set_title('inflow angle: '+str(-i-1)+'$^\circ$')
#         ax[j,i].legend(fontsize=13,loc=location[j])

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85,0.15, 0.02, 0.7])
# fig.colorbar(im, cax=cbar_ax,ticks=ticklevel, format='%.2f')
# plt.savefig('DEL_rotate_ref.png')


# name = ["rotate-3-0-0","rotate-3-25-15","rotate--3--20--20"]
# title = ["$0^\circ,0^\circ,0^\circ$","$-25^\circ,-15^\circ,0^\circ$","$20^\circ,20^\circ,0^\circ$"]
# levels = np.linspace(0.4,1.0,101)
# ticklevel = np.linspace(0.4,1.0,7)
# fig, ax = plt.subplots(1,3,figsize=(20, 4), dpi=300)
# for i in range(3):
#     f = h5py.File('../job/'+name[i]+'/output/'+name[i]+'_stat.h5','r')
#     inflow = np.mean(np.mean(f.get('u_avg')[50:100,:,44]))
#     print(inflow)

#     u = np.flip(np.array(f.get('u_avg')[:,:,44])/inflow,axis=1)
#     Iu = np.array(f.get('u_std')[:,:,44])
#     im = ax[i].contourf(u.T,levels,extent=[0,4.096,0,2.048],vmin=0.4,vmax=1,extend='both')
#     ax[i].axis('scaled')
#     ax[i].set_xlabel('x (km)')
#     ax[i].set_ylabel('y (km)')
#     ax[i].set_ylabel('y (km)')
#     ax[i].set_title(title[i])
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85,0.15, 0.02, 0.7])
#     fig.colorbar(im, cax=cbar_ax,ticks=ticklevel, format='%.2f')

# plt.savefig('mean_u_ref_rotate.png')




# levels = np.linspace(0.06,0.18,101)
# ticklevel = np.linspace(0.06,0.18,8)
# fig, ax = plt.subplots(1,3,figsize=(20, 4), dpi=300)
# for i in range(3):
#     f = h5py.File('../job/'+name[i]+'/output/'+name[i]+'_stat.h5','r')
#     inflow = np.mean(np.mean(f.get('u_avg')[50:100,:,44]))
#     print(inflow)

#     Iu = np.flip(np.array(f.get('u_std')[:,:,44]),axis=1)/inflow
#     im = ax[i].contourf(Iu.T,levels,extent=[0,4.096,0,2.048],vmin=0.06,vmax=0.18,extend='both')
#     ax[i].axis('scaled')
#     ax[i].set_xlabel('x (km)')
#     ax[i].set_ylabel('y (km)')
#     ax[i].set_ylabel('y (km)')
#     ax[i].set_title(title[i])
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85,0.15, 0.02, 0.7])
#     fig.colorbar(im, cax=cbar_ax,ticks=ticklevel, format='%.2f')
# plt.savefig('mean_uu_ref_rotate.png')



# mean_DEL = np.mean(DEL[:,:,5,:,0],axis=2)
# mean_DEL = np.sum(mean_DEL,axis=1)
# print(mean_DEL)
# plt.figure()
# plt.plot(mean_DEL)
# plt.ylim()
# plt.savefig('test.png')

# DEL_ref = mean_DEL[6,6,5,:]/mean_DEL[6,6,5,0]
# DEL_opti_positive = mean_DEL[i[0],j[0],5,:]/mean_DEL[6,6,5,0]
# DEL_opti_negative = mean_DEL[i2[0],j2[0],5,:]/mean_DEL[6,6,5,0]
# wt_name=['wt1','wt2','wt3']
# x = np.asarray([0,1,2])
# width = 0.3 
# print(DEL_ref)
# print(DEL_opti_positive)
# plt.figure()
# plt.bar(x-width,DEL_ref,width,label='$0^\circ,0^\circ$')
# plt.bar(x,DEL_opti_positive,width,label='$25^\circ,15^\circ$')
# plt.bar(x+width,DEL_opti_negative,width,label='$-25^\circ,-15^\circ$')
# plt.xticks(x,wt_name)
# plt.ylabel('Normalised DEL')
# plt.legend()
# plt.savefig('DEL_bar_ref.png')




# DEL_reference = total_DEL[6,6,0]
# normal_DEL = factor*(total_DEL[:,:,0]/DEL_reference-1)
# max_DEL = np.max(np.max(np.max(normal_DEL)))
# i,j = np.where(normal_DEL == max_DEL)
# levels = np.linspace(-climit,climit,101)
# ticklevel = np.linspace(-climit,climit,13)
# plt.contourf(normal_DEL.T,levels,extent=[-30,30,-30,30],vmax=climit,vmin=-climit,cmap='bwr', extend='both')
# plt.xlabel('WT1')
# plt.ylabel('WT2')
# max_name = '$('+str((i[0]-6)*5)+'^\circ'+','+str((j[0]-6)*5)+'^\circ'+')$: power gain '+str(round(max_DEL*100,2))+'%'
# print(max_name)
# # plt.text((i[0]-6)*5-6,(j[0]-6)*5+2,max_name,fontsize=10)
# plt.colorbar(ticks=ticklevel, format='%.2f')
# plt.clim(-climit,climit)
# if (flag==1):
#     plt.plot((i[0]-6)*5,(j[0]-6)*5,'o',label=max_name)
#     plt.legend(fontsize=12)
# plt.title(title_name,fontsize=22)
# # plt.title('Max power gain: '+str(round(max_DEL*100,2))+'%',fontsize=22)
# plt.savefig(name)