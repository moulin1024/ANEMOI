import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import animation, rc
import fatpack

def Goodman_method_correction(M_a,M_m,M_max):
    M_u = 1.5*M_max
    M_ar = M_a/(1-M_m/M_u)
    return M_ar
def get_DEL(m_f):
    m = 10
    Neq = 1000
    bins_num = 21
    bins_max = 10
    bins = np.linspace(0, bins_max, bins_num)
    bin_width = bins_max/(bins_num-1)
    bins_fine = np.linspace(0, bins_max, 501)

    rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=0.1, k=256)
    ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
    ranges_corrected = Goodman_method_correction(ranges,means,np.max(m_f))
    N, S = fatpack.find_range_count(ranges_corrected,bins)
    DEL = (np.sum(N*S**m)/Neq)**(1/m)
    return DEL

# flowfield   = np.load('flowfield.npy')
DEL_0 = np.load('DEL_0.npy')
DEL_1 = np.load('DEL_1.npy')
DEL_2 = np.load('DEL_2.npy')
DEL_3 = np.load('DEL_3.npy')
DEL_4 = np.load('DEL_4.npy')
DEL_5 = np.load('DEL_5.npy')
DEL_m1 = np.load('DEL_m1.npy')
DEL_m2 = np.load('DEL_m2.npy')
DEL_m3 = np.load('DEL_m3.npy')
DEL_m4 = np.load('DEL_m4.npy')
DEL_m5 = np.load('DEL_m5.npy')

print(DEL_0.shape)
power_0       = np.load('power_rotate_0.npy')
power_1       = np.load('power_rotate_1.npy')
power_2       = np.load('power_rotate_2.npy')
power_3       = np.load('power_rotate_3.npy')
power_4       = np.load('power_rotate_4.npy')
power_5       = np.load('power_rotate_5.npy')
power_m1       = np.load('power_rotate_-1.npy')
power_m2       = np.load('power_rotate_-2.npy')
power_m3       = np.load('power_rotate_-3.npy')
power_m4       = np.load('power_rotate_-4.npy')
power_m5       = np.load('power_rotate_-5.npy')
time        = np.load('time.npy')

# print(power.shape)

# DEL = np.zeros((13,13,1,3))
# for i in range(13):
#     print(i)
#     for j in range(13):
#         for k in range(1):
#             for l in range(3):
#                 DEL[i,j,k,l] = get_DEL(root_moment[i,j,k,l,0,10000:])

# x = np.linspace(1,512,512)*8
# y = np.linspace(1,256,256)*8

# def powermap(power,name):
#     plt.figure()
#     plt.rcParams.update({'font.size': 32})
#     total_power = np.sum(power,axis=3)
#     power_reference = total_power[6,6,0]
#     print(np.shape(total_power[:,:,0]))
#     plt.imshow(total_power[:,:,0]/power_reference-1,origin='lower',extent=[-32.5,32.5,-32.5,32.5])
#     plt.xlabel('WT1')
#     plt.ylabel('WT2')
#     plt.colorbar()
#     plt.savefig(name)
#     plt.colorbar()

def DELmap(DEL,name,climit,factor,title_name,flag):
    plt.figure()
    total_DEL = np.sum(DEL,axis=3)
    plt.rcParams.update({'font.size': 14})
    DEL_reference = total_DEL[6,6,0]
    normal_DEL = factor*(total_DEL[:,:,0]/DEL_reference-1)
    max_DEL = np.max(np.max(np.max(normal_DEL)))
    i,j = np.where(normal_DEL == max_DEL)
    levels = np.linspace(-climit,climit,101)
    ticklevel = np.linspace(-climit,climit,13)
    plt.contourf(normal_DEL.T,levels,extent=[-30,30,-30,30],vmax=climit,vmin=-climit,cmap='bwr', extend='both')
    plt.xlabel('WT1')
    plt.ylabel('WT2')
    max_name = '$('+str((i[0]-6)*5)+'^\circ'+','+str((j[0]-6)*5)+'^\circ'+')$: power gain '+str(round(max_DEL*100,2))+'%'
    print(max_name)
    # plt.text((i[0]-6)*5-6,(j[0]-6)*5+2,max_name,fontsize=10)
    plt.colorbar(ticks=ticklevel, format='%.2f')
    plt.clim(-climit,climit)
    if (flag==1):
        plt.plot((i[0]-6)*5,(j[0]-6)*5,'o',label=max_name)
        plt.legend(fontsize=12)
    plt.title(title_name,fontsize=14)
    # plt.title('Max power gain: '+str(round(max_DEL*100,2))+'%',fontsize=14)
    plt.savefig(name)


wt1 = np.linspace(-30,30,13)
wt2 = np.linspace(-30,30,13)
wt1_mesh,wt2_mesh = np.meshgrid(wt1,wt2,indexing='ij')

# def decision(power,DEL,name):
#     plt.figure()
#     wt1 = np.linspace(-30,30,13)
#     wt2 = np.linspace(-30,30,13)
#     wt1_mesh,wt2_mesh = np.meshgrid(wt1,wt2,indexing='ij')
#     total_power = np.sum(power,axis=3)
#     power_reference = total_power[6,6,0]
#     total_DEL = np.sum(DEL,axis=3)
#     DEL_reference = total_DEL[6,6,0]
#     normal_power = total_power[:,:,0]/total_power[6,6,0]-1
#     normal_DEL = total_DEL[:,:,0]/total_DEL[6,6,0]-1
#     plt.quiver(normal_power.flatten(),normal_DEL.flatten(),wt1_mesh.flatten(),wt2_mesh.flatten(),width=0.002)
#     # plt.plot(normal_power.flatten(),normal_DEL.flatten(),'o')
#     plt.xlim([-0.25,0.25])
#     plt.ylim([-0.25,0.25])
#     plt.savefig(name)

def decision(power,DEL,name):
    plt.figure()
    wt1 = np.linspace(-30,30,13)
    wt2 = np.linspace(-30,30,13)
    wt1_mesh,wt2_mesh = np.meshgrid(wt1,wt2,indexing='ij')
    total_power = np.sum(power,axis=3)
    power_reference = total_power[6,6,0]
    total_DEL = np.sum(DEL,axis=3)
    DEL_reference = total_DEL[6,6,0]
    normal_power = total_power[:,:,0]/total_power[6,6,0]-1
    normal_DEL = total_DEL[:,:,0]/total_DEL[6,6,0]-1
    # plt.quiver(normal_power.flatten(),normal_DEL.flatten(),wt1_mesh.flatten(),wt2_mesh.flatten(),width=0.002)
    plt.quiver(wt1_mesh.flatten(),wt2_mesh.flatten(),normal_power.flatten(),-normal_DEL.flatten(),width=0.003)
    # plt.plot(normal_power.flatten(),normal_DEL.flatten(),'o')
    # plt.xlim([-0.25,0.25])
    # plt.ylim([-0.25,0.25])
    plt.savefig(name)


def decision_new(power,DEL,name):
    plt.figure()
    wt1 = np.linspace(-30,30,13)
    wt2 = np.linspace(-30,30,13)
    wt1_mesh,wt2_mesh = np.meshgrid(wt1,wt2,indexing='ij')
    total_power = np.sum(power,axis=3)
    power_reference = total_power[6,6,0]
    total_DEL = np.sum(DEL,axis=3)
    DEL_reference = total_DEL[6,6,0]
    normal_power = total_power[:,:,0]/total_power[6,6,0]-1
    normal_DEL = total_DEL[:,:,0]/total_DEL[6,6,0]-1
    # plt.quiver(normal_power.flatten(),-normal_DEL.flatten(),wt1_mesh.flatten(),wt2_mesh.flatten(),width=0.002)
    # plt.quiver(wt1_mesh.flatten(),wt2_mesh.flatten(),normal_power.flatten(),-normal_DEL.flatten(),width=0.003)
    plt.plot(normal_power.flatten(),-normal_DEL.flatten(),'o')
    plt.xlim([-0.25,0.25])
    plt.ylim([-0.25,0.25])
    plt.savefig(name)

# plt.figure()
# power = power_0
# DEL = DEL_0

# total_power = np.sum(power,axis=3)
# power_reference = total_power[6,6,0]
# total_DEL = np.sum(DEL,axis=3)
# DEL_reference = total_DEL[6,6,0]
# normal_power = total_power[:,:,0]/total_power[6,6,0]-1
# normal_DEL = total_DEL[:,:,0]/total_DEL[6,6,0]-1
# plt.quiver(normal_power.flatten(),normal_DEL.flatten(),wt1_mesh.flatten(),wt2_mesh.flatten(),width=0.002)
# plt.xlim([-0.25,0.25])
# plt.ylim([-0.25,0.25])
# plt.savefig('post.png')
# DELmap()

# DELmap(power_m1,'power_m1.png',0.15,1,'inflow degree: $-1^\circ$',1)
# DELmap(power_m2,'power_m2.png',0.15,1,'inflow degree: $-2^\circ$',1)
# DELmap(power_m3,'power_m3.png',0.15,1,'inflow degree: $-3^\circ$',1)
# DELmap(power_m4,'power_m4.png',0.15,1,'inflow degree: $-4^\circ$',1)
# DELmap(power_m5,'power_m5.png',0.15,1,'inflow degree: $-5^\circ$',1)
# DELmap(power_0,'power_0.png',0.15,1,'inflow degree: $0^\circ$',1)
# DELmap(power_1,'power_1.png',0.15,1,'inflow degree: $1^\circ$',1)
# DELmap(power_2,'power_2.png',0.15,1,'inflow degree: $2^\circ$',1)
# DELmap(power_3,'power_3.png',0.15,1,'inflow degree: $3^\circ$',1)
# DELmap(power_4,'power_4.png',0.15,1,'inflow degree: $4^\circ$',1)
# DELmap(power_5,'power_5.png',0.15,1,'inflow degree: $5^\circ$',1)


# DELmap(DEL_0,'DEL_0.png',0.2,-1,'inflow degree: $0^\circ$',0)
# DELmap(DEL_1,'DEL_1.png',0.2,-1,'inflow degree: $1^\circ$',0)
# DELmap(DEL_2,'DEL_2.png',0.2,-1,'inflow degree: $2^\circ$',0)
# DELmap(DEL_3,'DEL_3.png',0.2,-1,'inflow degree: $3^\circ$',0)
# DELmap(DEL_4,'DEL_4.png',0.2,-1,'inflow degree: $4^\circ$',0)
# DELmap(DEL_5,'DEL_5.png',0.2,-1,'inflow degree: $5^\circ$',0)
# DELmap(DEL_m1,'DEL_m1.png',0.2,-1,'inflow degree: $-1^\circ$',0)
# DELmap(DEL_m2,'DEL_m2.png',0.2,-1,'inflow degree: $-2^\circ$',0)
# DELmap(DEL_m3,'DEL_m3.png',0.2,-1,'inflow degree: $-3^\circ$',0)
# DELmap(DEL_m4,'DEL_m4.png',0.2,-1,'inflow degree: $-4^\circ$',0)
# DELmap(DEL_m5,'DEL_m5.png',0.2,-1,'inflow degree: $-5^\circ$',0)

# # plt.figure()
# decision(power_0,DEL_0,'decision_0.png')
# decision(power_1,DEL_1,'decision_1.png')
# decision(power_2,DEL_2,'decision_2.png')
# decision(power_3,DEL_3,'decision_3.png')
# decision(power_4,DEL_4,'decision_4.png')
# decision(power_5,DEL_5,'decision_5.png')
# decision(power_m1,DEL_m1,'decision_m1.png')
# decision(power_m2,DEL_m2,'decision_m2.png')
# decision(power_m3,DEL_m3,'decision_m3.png')
# decision(power_m4,DEL_m4,'decision_m4.png')
# decision(power_m5,DEL_m5,'decision_m5.png')

decision_new(power_0,DEL_0,'decision_new_0.png')
decision_new(power_1,DEL_1,'decision_new_1.png')
decision_new(power_2,DEL_2,'decision_new_2.png')
decision_new(power_3,DEL_3,'decision_new_3.png')
decision_new(power_4,DEL_4,'decision_new_4.png')
decision_new(power_5,DEL_5,'decision_new_5.png')
decision_new(power_m1,DEL_m1,'decision_new_m1.png')
decision_new(power_m2,DEL_m2,'decision_new_m2.png')
decision_new(power_m3,DEL_m3,'decision_new_m3.png')
decision_new(power_m4,DEL_m4,'decision_new_m4.png')
decision_new(power_m5,DEL_m5,'decision_new_m5.png')


# decision_new(power_0,DEL_0,'decision_new_0.png')
# decision_new(power_1,DEL_1,'decision_new_1.png')
# decision_new(power_3,DEL_3,'decision_new_3.png')
# decision_new(power_4,DEL_4,'decision_new_4.png')
# decision_new(power_5,DEL_5,'decision_new_5.png')
# decision_new(power_m3,DEL_m3,'decision_new_m3.png')
# decision_new(power_m5,DEL_m5,'decision_new_m5.png')
# plt.savefig('total_decision.png')
# print(total_power)

# plt.figure(figsize=(12, 6), dpi=80)
# plt.plot(time,root_moment[0,0,0,0,0,:],lw=1,alpha=0.5)
# plt.savefig('series.png')

# DEL = np.zeros((4,4,3))
# for i in range(4):
#     for j in range(4):
#         for k in range(3):
#             DEL[i,j,k] = get_DEL(root_moment[i,j,k,0,10000:])

# wt_name = ['WT1','WT2','WT3']
# fig = figure(figsize=(10,8))
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)
# def animate(case):  
#     ax1.cla()
#     col = 0
#     row = 0
#     ax1.contourf(x,y,flowfield[row,col,case,0,:,:].T,100,vmin=5,vmax=12)
#     ax1.set_xlabel('x (m)')
#     ax1.set_ylabel('y (m)')
#     ax1.set_title('$\overline{u}$: inflow '+str(case)+' degree')
#     ax1.axis('scaled')

#     ax2.cla()
#     ax2.contourf(x,y,flowfield[row,col,case,3,:,:].T/11,100,vmin=0.05,vmax=0.18)
#     ax2.set_xlabel('x (m)')
#     ax2.set_ylabel('y (m)')
#     ax2.set_title('$I_u$: inflow '+str(case)+' degree')
#     ax2.axis('scaled')

#     ax3.cla()
#     ax3.bar(wt_name,power[row,col,case,:]/power_reference)
#     ax3.set_ylim([0,1.1])
#     ax3.set_title('Normlised power')

#     ax4.cla()
#     ax4.bar(wt_name,DEL[row,col,case,:]/DEL_reference)
#     ax4.set_ylim([0,1.5])
#     ax4.set_title('Normlised blade flapwise DEL')
#     print(row,col)
#     return
# anim = animation.FuncAnimation(fig, animate, frames=6)
# anim.save('animation_inflow.gif',writer='pillow', fps=1)


# total_DEL = np.sum(DEL,axis=3)
# total_power = np.sum(power,axis=3)

# print(total_power)


# for i in range(1):
#     fig,ax = plt.subplots(1,3,figsize=(24, 6),dpi=100)
#     plt.rcParams.update({'font.size': 16})
#     normal_power = total_power[:,:,i]/total_power[6,6,i]-1
#     normal_DEL = total_DEL[:,:,i]/total_DEL[6,6,i]-1
#     ax[0].cla()
#     im0 = ax[0].imshow(normal_power.T,origin='lower',cmap='bwr',vmin=-0.2,vmax=0.2,extent=[-30,30,-30,30])
#     ax[0].set_xlabel('$\gamma_1$ (degree)',fontsize=16)
#     ax[0].set_ylabel('$\gamma_2$ (degree)',fontsize=16)
#     plt.colorbar(im0,ax=ax[0])
#     ax[0].set_title('$\Delta P/P_{0}$')
#     ax[1].cla()
#     im1 = ax[1].imshow(-normal_DEL.T,origin='lower',cmap='bwr',vmin=-0.2,vmax=0.2,extent=[-30,30,-30,30])
#     ax[1].set_xlabel('$\gamma_1$ (degree)',fontsize=16)
#     ax[1].set_ylabel('$\gamma_2$ (degree)',fontsize=16)
#     ax[1].set_title('$-\Delta DEL/DEL_{0}$')
#     # plt.colorbar(im1,ax=ax[1])
#     ax[2].cla()
#     ax[2].plot(normal_power.flatten(),-normal_DEL.flatten(),'o')
#     ax[2].set_xlim([-0.2,0.2])
#     ax[2].set_ylim([-0.2,0.2])
#     ax[2].set_xlabel('$\Delta P/P_{0}$',fontsize=16)
#     ax[2].set_ylabel('$\Delta DEL/DEL_{0}$',fontsize=16)
#     plt.savefig('decision'+'_'+str(i)+'.png')




# m = 10
# Neq = 1000
# bins_num = 51
# bins_max = 10
# bins = np.linspace(0, bins_max, bins_num)
# bin_width = bins_max/(bins_num-1)
# bins_fine = np.linspace(0, bins_max, 501)


# plt.figure()

# m_f = root_moment[0,0,0,0,:]
# print(get_DEL(m_f))
# rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=0.1, k=256)
# ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
# ranges_corrected = Goodman_method_correction(ranges,means,np.max(m_f))
# N, S = fatpack.find_range_count(ranges_corrected,bins)
# Ncum = N.sum() - np.cumsum(N)
# damage = np.cumsum(N*S**m)/np.sum(N*S**m)
# plt.plot(S,damage)

# m_f = root_moment[0,0,1,0,:]
# print(get_DEL(m_f))
# rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=0.1, k=256)
# ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
# ranges_corrected = Goodman_method_correction(ranges,means,np.max(m_f))
# N, S = fatpack.find_range_count(ranges_corrected,bins)
# Ncum = N.sum() - np.cumsum(N)
# damage = np.cumsum(N*S**m)/np.sum(N*S**m)
# plt.plot(S,damage)

# m_f = root_moment[0,0,2,0,:]
# print(get_DEL(m_f))
# rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=0.1, k=256)
# ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
# ranges_corrected = Goodman_method_correction(ranges,means,np.max(m_f))
# N, S = fatpack.find_range_count(ranges_corrected,bins)
# Ncum = N.sum() - np.cumsum(N)
# damage = np.cumsum(N*S**m)/np.sum(N*S**m)
# plt.plot(S,damage)

# plt.savefig('range_histgram.png')
# # data = root_moment[2,0,2,0,:]-np.mean(root_moment[2,0,2,0,:])
# # ps = np.abs(np.fft.fft(data))**2
# # time_step = 0.02
# # freqs = np.fft.fftfreq(data.size, time_step)
# # idx = np.argsort(freqs)
# # plt.figure(figsize=(12, 6), dpi=80)
# # plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='Flapwise')
# # plt.ylim(1e-4,1e10)


# # data = root_moment[2,1,2,0,:]-np.mean(root_moment[2,1,2,0,:])
# # ps = np.abs(np.fft.fft(data))**2
# # time_step = 0.02
# # freqs = np.fft.fftfreq(data.size, time_step)
# # idx = np.argsort(freqs)
# # # plt.figure(figsize=(12, 6), dpi=80)
# # plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='Flapwise')
# # plt.ylim(1e-4,1e10)

# # data = root_moment[2,2,2,0,:]-np.mean(root_moment[2,2,2,0,:])
# # ps = np.abs(np.fft.fft(data))**2
# # time_step = 0.02
# # freqs = np.fft.fftfreq(data.size, time_step)
# # idx = np.argsort(freqs)
# # # plt.figure(figsize=(12, 6), dpi=80)
# # plt.loglog(freqs[idx], ps[idx],alpha=0.5,label='Flapwise')
# # plt.ylim(1e-4,1e10)

# # plt.savefig('psd.png')
