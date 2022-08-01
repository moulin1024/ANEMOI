import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
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
    bins_num = 51
    bins_max = 20
    bins = np.linspace(0, bins_max, bins_num)
    bin_width = bins_max/(bins_num-1)

    rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=0.1, k=256)
    ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
    ranges_corrected = Goodman_method_correction(ranges,means,np.max(m_f))
    N, S = fatpack.find_range_count(ranges_corrected,bins)
    DEL = (np.sum(N*S**m)/Neq)**(1/m)
    return DEL

# # flowfield   = np.load('flowfield.npy')
# root_moment_rotate_m5 = np.load('root_moment_rotate_m5.npy')
# root_moment_rotate_m4 = np.load('root_moment_rotate_m4.npy')
# root_moment_rotate_m3 = np.load('root_moment_rotate_m3.npy')
# root_moment_rotate_m2 = np.load('root_moment_rotate_m2.npy')
# root_moment_rotate_m1 = np.load('root_moment_rotate_m1.npy')
# root_moment_rotate_0 = np.load('root_moment_rotate_0.npy')
# root_moment_rotate_1 = np.load('root_moment_rotate_1.npy')
# root_moment_rotate_2 = np.load('root_moment_rotate_2.npy')
# root_moment_rotate_3 = np.load('root_moment_rotate_3.npy')
# root_moment_rotate_4 = np.load('root_moment_rotate_4.npy')
# root_moment_rotate_5 = np.load('root_moment_rotate_5.npy')

time        = np.load('time.npy')

# print(root_moment.shape)
# root_moment = np.zeros((13,13,11,3,2,49001))
DEL_flap = np.zeros((13,13,5,3,3))
DEL_edge = np.zeros((13,13,5,3,3))
# DEL_m4 = np.zeros((13,13,1,3))
# DEL_m3 = np.zeros((13,13,1,3))
# DEL_m2 = np.zeros((13,13,1,3))
# DEL_m1 = np.zeros((13,13,1,3))
# DEL_0 = np.zeros((13,13,1,3))
# DEL_1 = np.zeros((13,13,1,3))
# DEL_2 = np.zeros((13,13,1,3))
# DEL_3 = np.zeros((13,13,1,3))
# DEL_4 = np.zeros((13,13,1,3))
# DEL_5 = np.zeros((13,13,1,3))
inflow = [-5,-3,0,3,5]
for i in range(13):
    print(i)
    for j in range(13):
        for k in range(5):
            name = "rotate-"+str(inflow[k])+'-'+str((i-6)*5)+"-"+str((j-6)*5)
            print(name)
            f = h5py.File('../job/'+name+'/output/'+name+'_force.h5','r')
            for wt in range(3):
                for blade in range(3):
                    flap_moment = np.array(f.get('moment_flap')[10000:,blade,0,wt])/10**6
                    edge_moment = np.array(f.get('moment_edge')[10000:,blade,1,wt])/10**6
                    DEL_flap[i,j,k,blade,wt] = get_DEL(flap_moment)
                    DEL_edge[i,j,k,blade,wt] = get_DEL(edge_moment)


np.save('DEL_flap.npy', DEL_flap)
np.save('DEL_edge.npy', DEL_edge)


# np.save('DEL_m4.npy', DEL_m4)
# np.save('DEL_m3.npy', DEL_m3)
# np.save('DEL_m2.npy', DEL_m2)
# np.save('DEL_m1.npy', DEL_m1)
# np.save('DEL_0.npy', DEL_0)
# np.save('DEL_1.npy', DEL_1)
# np.save('DEL_2.npy', DEL_2)
# np.save('DEL_3.npy', DEL_3)
# np.save('DEL_4.npy', DEL_4)
# np.save('DEL_5.npy', DEL_5)
# x = np.linspace(1,512,512)*8
# y = np.linspace(1,256,256)*8

# def DELmap(DEL,name):
#     plt.figure()
#     total_DEL = DEL[:,:,0,0]#np.sum(DEL,axis=3)
#     DEL_reference = total_DEL[6,6,0]
#     DEL_max = np.max(np.max(total_DEL[:,:,0]/DEL_reference-1))
#     DEL_min = np.min(np.min(total_DEL[:,:,0]/DEL_reference-1))
#     climit = np.max([np.abs(DEL_max),np.abs(DEL_min)])
#     print(np.shape(total_DEL[:,:,0]))
#     plt.imshow(total_DEL[:,:,0]/DEL_reference-1,origin='lower',extent=[-32.5,32.5,-32.5,32.5],vmin=-climit,vmax=climit,cmap='bwr')
#     plt.xlabel('WT1')
#     plt.ylabel('WT2')
#     plt.colorbar()
#     plt.savefig(name)
#     plt.colorbar()

# DELmap(DEL_0,'DEL_0.png')
# powermap(power_m5,'power_m5.png')
# powermap(power_3,'power_3.png')
# powermap(power_0,'power_0.png')
# powermap(power_5,'power_5.png')

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
