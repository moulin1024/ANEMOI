import numpy as np
import matplotlib.pyplot as plt 
import fatpack
import h5py

def Goodman_method_correction(M_a,M_m,M_max):
    M_u = 1.5*M_max
    M_ar = M_a/(1-M_m/M_u)
    return M_ar

def get_DEL(root_moment):
    m = 10
    Neq = 1e6
    bins_num = 51
    bins_max = 15
    bins = np.linspace(0, bins_max, bins_num)
    bin_width = bins_max/(bins_num-1)

    N_all = np.zeros([bins_num-1,3])
    S_all = np.zeros([bins_num-1,3])
    DEL_avg = np.zeros(8)
    for j in range(8):
        for i in range(3):
            m_f = root_moment[:,i,j]/1e6
            rev, rev_ix = fatpack.find_reversals_racetrack_filtered(m_f, h=0.1, k=256)
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

f = h5py.File('../job/dyn-yaw-8wt-baseline/output/dyn-yaw-8wt-baseline_force.h5','r')

def multiply_along_axis(A, B, axis):
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)

phase = f['phase'][901:,0,:,:]

time = f['time'][901:]
root_moment_flap = f['moment_flap'][901:,:,0,:]
# root_moment_edge = f['moment_edge'][901:,:,0,:]
yaw_moment = np.sum(np.cos(phase)*root_moment_flap,axis=1)
DEL_baseline = np.zeros([8,2])
DEL_baseline[:,0] = get_DEL(root_moment_flap)
DEL_baseline[:,1] = get_DEL_nac(yaw_moment)

# print(yaw_moment.shape)

DEL_dyn = np.zeros([3,4,8,2])
yaw_mag_list = [10,20,30]
yaw_period_list = [300,400,500,600]
for ii,yaw_mag in enumerate(yaw_mag_list):
    print(yaw_mag)
    for jj,yaw_period in enumerate(yaw_period_list):
        f = h5py.File('../job/dyn-yaw-8wt-'+str(yaw_mag)+'-'+str(yaw_period)+'s/output/dyn-yaw-8wt-'+str(yaw_mag)+'-'+str(yaw_period)+'s_force.h5','r')
        phase = f['phase'][901:,0,:,:]
        time = f['time'][901:]
        root_moment_flap = f['moment_flap'][901:,:,0,:]
        yaw_moment = np.sum(np.cos(phase)*root_moment_flap,axis=1)
        DEL_dyn[ii,jj,:,0] = get_DEL(root_moment_flap)
        DEL_dyn[ii,jj,:,1] = get_DEL_nac(yaw_moment)


DEL_static = np.zeros([3,8,2])
yaw_mag_list = [10,20,30]
for ii,yaw_mag in enumerate(yaw_mag_list):
    print(yaw_mag)
    f = h5py.File('../job/dyn-yaw-8wt-'+str(yaw_mag)+'-static/output/dyn-yaw-8wt-'+str(yaw_mag)+'-static_force.h5','r')
    phase = f['phase'][901:,0,:,:]
    time = f['time'][901:]
    root_moment_flap = f['moment_flap'][901:,:,0,:]
    yaw_moment = np.sum(np.cos(phase)*root_moment_flap,axis=1)
    DEL_static[ii,:,0] = get_DEL(root_moment_flap)
    DEL_static[ii,:,1] = get_DEL_nac(yaw_moment)

np.savez('DEL_results.npz',DEL_baseline=DEL_baseline,DEL_dyn=DEL_dyn,DEL_static=DEL_static)