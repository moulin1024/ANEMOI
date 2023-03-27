import numpy as np
import matplotlib.pyplot as plt 
import fatpack
import h5py

def Goodman_method_correction(M_a,M_m,M_max):
    M_u = 1.5*M_max
    M_ar = M_a/(1-M_m/M_u)
    return M_ar

def get_del(root_moment):
    m = 10
    Neq = 1e6
    bins_num = 101
    bins_max = 30
    bins = np.linspace(0, bins_max, bins_num)
    bin_width = bins_max/(bins_num-1)

    N_all = np.zeros([bins_num-1,3])
    S_all = np.zeros([bins_num-1,3])
    DEL_avg = np.zeros(8)
    for j in range(8):
        for i in range(3):
            m_f = root_moment[:,i,j]/1e6
            rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=0, k=256)
            ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
            ranges_corrected = Goodman_method_correction(ranges,means,np.max(m_f))
            N, S = fatpack.find_range_count(ranges_corrected,bins)
            N_all[:,i] = N
            S_all[:,i] = S

        N_avg = np.mean(N_all,axis=1)
        S_avg = np.mean(S_all,axis=1)
        DEL_avg[j] = (np.sum(N_avg*S_avg**m)/Neq)**(1/m)

    return DEL_avg


def get_DEL_nac(yaw_moment):
    m = 10
    Neq = 1e6
    bins_num = 51
    bins_max = 15
    bins = np.linspace(0, bins_max, bins_num)
    bin_width = bins_max/(bins_num-1)

    DEL_avg = np.zeros(8)
    for j in range(8):
        m_f = yaw_moment[:,j]/1e6
        rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=0.1, k=256)
        ranges,means = fatpack.find_rainflow_ranges(rev, k=256, return_means=True)
        ranges_corrected = Goodman_method_correction(ranges,means,np.max(m_f))
        N, S = fatpack.find_range_count(ranges_corrected,bins)
        DEL_avg[j] = (np.sum(N*S**m)/Neq)**(1/m)

    return DEL_avg


nacelle_inertia = 2607890
rotor_mass = 110000
rotor_inertia = 0.25*(rotor_mass*63**2)+rotor_mass*5**2
total_inertia = nacelle_inertia + rotor_inertia


f = h5py.File('../job/dyn-yaw-8wt-baseline/output/dyn-yaw-8wt-baseline_force.h5','r')
phase = f['phase'][:,0,:,:]
time = f['time'][:]
root_moment_flap = f['moment_flap'][:,:,0,:]
yaw_moment = np.sum(np.cos(phase)*root_moment_flap,axis=1)
# acceleration = -np.pi/6*(2*np.pi/400)**2*np.sin(2*np.pi/400*time)
# inertia_moment = -acceleration*total_inertia
root_moment_flap = f['moment_flap'][:,:,0,:]
yaw_moment = np.sum(np.cos(phase)*root_moment_flap,axis=1)
print(yaw_moment.shape)
inertia_moment = np.zeros(yaw_moment.shape)
# inertia_moment[:,0] = -acceleration*total_inertia
total_yaw_moment = yaw_moment+inertia_moment
# plt.plot(time,root_moment_flap[:,0,0]/1e3,alpha=0.3)

plt.figure(figsize=(18,5),dpi=300)
plt.rcParams.update({'font.size': 14})
plt.plot(time,root_moment_flap[:,0,0]/1e3,label='flapwise root bending moment')
plt.plot(time,total_yaw_moment[:,0]/1e3,label='yaw moment')
plt.xlabel('time (s)')
plt.ylabel('moment (kN.m)')
plt.legend()
plt.ylim([-2000,10000])
plt.savefig('moment_baseline.png')

del_baseline= get_del(root_moment_flap)
del_baseline_nac= get_DEL_nac(total_yaw_moment)

f = h5py.File('../job/dyn-yaw-8wt-30-400s/output/dyn-yaw-8wt-30-400s_force.h5','r')
phase = f['phase'][:,0,:,:]
time = f['time'][:]
acceleration = -np.pi/6*(2*np.pi/400)**2*np.sin(2*np.pi/400*time)
inertia_moment = -acceleration*total_inertia
root_moment_flap = f['moment_flap'][:,:,0,:]
yaw_moment = np.sum(np.cos(phase)*root_moment_flap,axis=1)
inertia_moment = np.zeros(yaw_moment.shape)
inertia_moment[:,0] = -acceleration*total_inertia
total_yaw_moment = yaw_moment+inertia_moment
# plt.plot(time,root_moment_flap[:,0,0]/1e3,alpha=0.3)

# plt.xlim([0,100])
print(get_DEL_nac(total_yaw_moment))
del_30_400s= get_del(root_moment_flap)
del_30_400s_nac= get_DEL_nac(total_yaw_moment)

np.save('root_moment_flap_30.npy',root_moment_flap)

plt.figure(figsize=(18,5),dpi=300)
plt.rcParams.update({'font.size': 14})
plt.plot(time,root_moment_flap[:,0,0]/1e3,label='flapwise root bending moment')
plt.plot(time,total_yaw_moment[:,0]/1e3,label='yaw moment')
plt.xlabel('time (s)')
plt.ylabel('moment (kN.m)')
plt.legend()
plt.ylim([-2000,10000])
plt.savefig('moment_30_400s.png')

f = h5py.File('../job/dyn-yaw-8wt-10-100s/output/dyn-yaw-8wt-10-100s_force.h5','r')
phase = f['phase'][:,0,:,:]
time = f['time'][:]
root_moment_flap = f['moment_flap'][:,:,0,:]
yaw_moment = np.sum(np.cos(phase)*root_moment_flap,axis=1)
acceleration = -np.pi/18*(2*np.pi/100)**2*np.sin(2*np.pi/100*time)
inertia_moment = -acceleration*total_inertia
root_moment_flap = f['moment_flap'][:,:,0,:]
yaw_moment = np.sum(np.cos(phase)*root_moment_flap,axis=1)
inertia_moment = np.zeros(yaw_moment.shape)
inertia_moment[:,0] = -acceleration*total_inertia
total_yaw_moment = yaw_moment+inertia_moment

del_10_100s= get_del(root_moment_flap)
del_10_100s_nac= get_DEL_nac(total_yaw_moment)

reference = del_baseline[0]
reference_nac = del_baseline_nac[0]


np.save('root_moment_flap_10.npy',root_moment_flap)


plt.figure(figsize=(18,5),dpi=300)
plt.rcParams.update({'font.size': 14})
plt.plot(time,root_moment_flap[:,0,0]/1e3,label='flapwise root bending moment')
plt.plot(time,total_yaw_moment[:,0]/1e3,label='yaw moment')
plt.xlabel('time (s)')
plt.ylabel('moment (kN.m)')
plt.legend()
plt.ylim([-2000,10000])
plt.savefig('moment_10_100s.png')
barWidth = 0.25

plt.figure(figsize=(12,8),dpi=300)
plt.rcParams.update({'font.size': 24})
ticks = ['WT'+ str(x+1) for x in range(8)]
br1 = np.arange(8)
br2 = [x + barWidth for x in br1]
br3 = [x - barWidth for x in br1]
plt.bar(br3, del_baseline_nac/reference_nac, width = barWidth,edgecolor ='grey', label ='Zero-yaw baseline')
plt.bar(br1, del_30_400s_nac/reference_nac, width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 30^\circ$, period 400 s)')
plt.bar(br2, del_10_100s_nac/reference_nac, width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 10^\circ$, period 100 s')
plt.xticks(br1,ticks)
plt.ylabel('Normalised DEL')
plt.ylim([0,2.0])
plt.legend(fontsize=20)
plt.savefig('DEL-yaw.png')



plt.figure(figsize=(12,8),dpi=300)
plt.rcParams.update({'font.size': 24})
ticks = ['WT'+ str(x+1) for x in range(8)]
br1 = np.arange(8)
br2 = [x + barWidth for x in br1]
br3 = [x - barWidth for x in br1]
plt.bar(br3, del_baseline/reference, width = barWidth,edgecolor ='grey', label ='Zero-yaw baseline')
plt.bar(br1, del_30_400s/reference, width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 30^\circ$, period 400 s)')
plt.bar(br2, del_10_100s/reference, width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 10^\circ$, period 100 s')
plt.xticks(br1,ticks)
plt.ylabel('Normalised DEL')
plt.ylim([0,2.0])
plt.legend(fontsize=20)
plt.savefig('DEL-flap.png')

print(del_baseline)
print(del_30_400s)
print(del_10_100s)

print(del_baseline_nac)
print(del_30_400s_nac)
print(del_10_100s_nac)

print(np.sum(del_baseline))
print(np.sum(del_30_400s))
print(np.sum(del_10_100s))

print(np.sum(del_baseline_nac))
print(np.sum(del_30_400s_nac))
print(np.sum(del_10_100s_nac))
# print(total_inertia*np.pi/18*(2*np.pi/100)**2)
# plt.savefig('test.png')
# # plt.plot(time,root_moment_flap[:,0,0])


# # f = h5py.File('../job/dyn-yaw-8wt-30-400s/output/dyn-yaw-8wt-30-400s_force.h5','r')
# # phase = f['phase'][9001:,0,:,:]
# # time = f['time'][9001:]
# # root_moment_flap = f['moment_flap'][9001:,:,0,:]

# # Inertia = 223652890
# # # (2*3.14/100)^2*3.14/30*
# # inertia_force = -Inertia*np.pi/6*(2*np.pi/300)**2*np.sin(2*np.pi/300*time)
# # print(np.sum(np.cos(phase[0,:,0])*root_moment_flap[0,:,0]))
# # print(root_moment_flap[0,:,0])
# # yaw_moment = np.sum(np.cos(phase)*root_moment_flap,axis=1)
# # # print(yaw_moment[0,0])


# # plt.plot(time,yaw_moment[:,0])
# plt.savefig('test.png')