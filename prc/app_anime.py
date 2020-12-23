'''
Created on 02.07.2019

@author: Mou Lin (moulin1024@gmail.com)

--------------------------------------------------------------------------------
app: create self-explained hdf5 output file
--------------------------------------------------------------------------------
'''

################################################################################
# IMPORT
################################################################################
import os, sys
import fctlib
import app_post as post
import numpy as np
import h5py
import matplotlib.pyplot as plt
import h5py
import os.path
import math
import pandas as pd
from pathlib import Path
from matplotlib.pyplot import figure
from matplotlib import animation, rc
# from pyevtk.hl import gridToVTK


################################################################################
# MAIN FONCTION
################################################################################
def anime(PATH, case_name):
    '''
    DEF:    post-processing for wireles.
    INPUT:  - case_name
    OUTPUT: - Statistics field: stat.h5
            - Instantanous field: animation.h5 
    '''
    case_path = fctlib.get_case_path(PATH, case_name)

    ############################################################################
    # INIT
    out_path = os.path.join(case_path, 'output')
    in_path = os.path.join(case_path, 'input')
    fctlib.test_and_mkdir(out_path)
    src_out_path = os.path.join(PATH['job'], case_name, 'src', 'output')
    src_inp_path = os.path.join(PATH['job'], case_name, 'src', 'input')

    ############################################################################
    # CONFIG
    print('extract config...')
    config = fctlib.get_config(case_path)

    ############################################################################
    # COMPUTE
    print('compute results...')
    print('Flow Fields:')
    print('Flow Fields:')

    space = post.get_space(config)
    time = post.get_time(config)

    if config['ta_flag'] > 0:
        result_3d = post.get_result_3d(src_inp_path, src_out_path, config)

        f = h5py.File(out_path+'/'+case_name+'_ta.h5', 'w')
        grp = f.create_group("data")
        # print(result_3d['u_avg_c'].shape)
        dset = grp.create_dataset("u_avg", data=result_3d['u_avg_c'])
        dset = grp.create_dataset("v_avg", data=result_3d['v_avg_c'])
        dset = grp.create_dataset("w_avg", data=result_3d['w_avg_c'])
        dset = grp.create_dataset("u_std", data=result_3d['u_std_c'])
        dset = grp.create_dataset("v_std", data=result_3d['v_std_c'])
        dset = grp.create_dataset("w_std", data=result_3d['w_std_c'])

        x_grid_unmask = space['x']
        y_grid_unmask = space['y']
        z_grid_unmask = space['z_c']

        x = x_grid_unmask[config['ts_istart']-1:config['ts_iend']]
        y = y_grid_unmask[config['ts_jstart']-1:config['ts_jend']]
        z = z_grid_unmask[:config['ts_kend']-1]

        dset = grp.create_dataset("x", data=x)
        dset = grp.create_dataset("y", data=y)
        dset = grp.create_dataset("z", data=z)

    # gridToVTK(
    #     out_path+'/'+case_name,
    #     x,
    #     y,
    #     z,
    #     pointData={"u": result_3d['u_avg_c'],"v": result_3d['v_avg_c'],"w": result_3d['w_avg_c']}
    # )

        for key, value in config.items():
            grp.attrs[key]=value

        f.close()
        if config['turb_flag'] > 0:
            df = pd.read_csv(in_path+'/turb_loc.dat')
            df_power = pd.read_csv(src_out_path+'/ta_power.dat',header=None)
            df['power'] = df_power
            print(df['power'])
            print(np.sum(df_power.to_numpy()))
            df.to_csv(out_path+'/ta_power.csv',index=False)


    #         # Spatial coorindates (masked)


    #     XX,YY = np.meshgrid(x,y,indexing='ij')
    #     YY2,ZZ = np.meshgrid(y,z,indexing='ij')
    #     u = result_3d['u_avg_c']
    #     v = result_3d['v_avg_c']
    #     w = result_3d['w_avg_c']

    #     # result_pr = post.get_result_pr(result_3d, config)
    # if config['ts_flag'] > 0:
    #     result_4d = post.get_result_4d(src_out_path, config)

    # f = h5py.File(out_path+'/'+case_name+'_ts.h5', 'w')

    hub_k = int(config['turb_z']/config['dz'])+1
    print(hub_k)

    # u2 = u*u - np.mean(u,axis=0)*np.mean(u,axis=0)
    # v2 = v*v - np.mean(v,axis=0)*np.mean(v,axis=0)
    # w2 = w*w - np.mean(w,axis=0)*np.mean(w,axis=0)
    # uw = u*w - np.mean(u,axis=0)*np.mean(w,axis=0)
    
    # print(u.shape)
    # def rms_profile(x2):
    #     x2_mean = np.sqrt(np.mean(np.mean(np.mean(x2[:,:,:,:],axis=0),axis=0),axis=0))
    #     return x2_mean
    
    # def vel_profile(x):
    #     x_mean = np.mean(np.mean(np.mean(x[:,:,:,:],axis=0),axis=0),axis=0)
    #     return x_mean
    
    # u_profile = vel_profile(u)
    # v_profile = vel_profile(v)
    # w_profile = vel_profile(w)

    # u2_profile = rms_profile(u2)
    # v2_profile = rms_profile(v2)
    # w2_profile = rms_profile(w2)

    # print(u_profile[hub_k])
    # print(u2_profile[hub_k]/u_profile[hub_k])
    # # u_hori = np.sqrt(u**2 + v**2)
    # # velocity profile
    # figure(num=None, figsize=(12, 3), dpi=100, facecolor='w', edgecolor='k')
    # log_profle = 1.0/0.4*np.log((z/config['zo']))
    # # log_profle = 0.25/0.4*np.log((z/(config['zo'])))
    # plt.semilogx(z/config['lz'],u_profile,'o')
    # plt.semilogx(z/config['lz'],log_profle,'k--')
    # plt.subplot(141)
    # plt.plot(u_profile,z/config['lz'],label='sp')
    # plt.xlabel('$\overline{u}/u_*$')
    # plt.ylabel('$z/H$')
    # plt.legend()

    # plt.subplot(142)
    # plt.plot(u2_profile,z/config['lz'])
    # plt.xlabel('$\sigma_u/u_*$')
    # plt.ylabel('$z/H$')

    # plt.subplot(143)
    # plt.plot(v2_profile,z/config['lz'])
    # plt.xlabel('$\sigma_v/u_*$')
    # plt.ylabel('$z/H$')

    # plt.subplot(144)
    # plt.plot(w2_profile,z/config['lz'])
    # plt.xlabel('$\sigma_w/u_*$')
    # plt.ylabel('$z/H$')

    # plt.tight_layout()

    # plt.savefig('profile.png')
    # # np.save('u_sp',mean_profile)


    # # # Mean field at hub-height

    # figure(num=None, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
    # u_mean = np.mean(u,axis=0)
    # u2_mean = np.mean(u*u,axis=0) - u_mean*u_mean
    # u_mean_field = u_mean
    # # plt.plot(u_mean_field[156,:,20],'.')
    # # plt.ylim([0,-0.8])
    # # plt.plot(np.mean(np.mean(u2_mean,axis=0),axis=0))
    # # u_mean_field = (u_mean/5.2)-1
    # plt.imshow(u2_mean[:,:,hub_k].T,origin='lower',aspect=config['dy']/config['dx'])
    # plt.colorbar()
    # # plt.clim(0,0.2)
    # # plt.plot(u2_mean[:,14,20])
    # plt.savefig('field.png')
    # # print('Print mean field')
    # # np.save('u_mean.npy',u_mean)
    # # np.save('u2_mean.npy',u2_mean)

    if config['ts_flag'] > 0:
        result_4d = post.get_result_4d(src_out_path, config)

        u = result_4d['u_inst_c']
        v = result_4d['v_inst_c']
        w = result_4d['w_inst_c']

        fig, ax = plt.subplots(2,1)
        # i = 9
        def animate(i):    #     azimuths = np.radians(np.linspace(0, 360, 40))
        #     zeniths = np.linspace(0, 0.5, 30)
        #     theta,r = np.meshgrid(azimuths,zeniths,indexing='ij')
            values = u[i,:,:,hub_k]#np.random.random((azimuths.size, zeniths.size))
            im1 = ax[0].imshow(values.T,origin='lower',aspect=config['dy']/config['dx'])
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            values = u[i,64,:,:]#np.random.random((azimuths.size, zeniths.size))
            im2 = ax[1].imshow(values.T,origin='lower',aspect=config['dz']/config['dy'])
            # im2 = ax[1].quiver(v[i,300,1::4,1::4].T,w[i,300,1::4,1::4].T,scale=10)
            # ax[1].scatter([63],[19],marker='+',color='r')
            ax[1].set_xlabel('y')
            ax[1].set_ylabel('z')
            print(i)
            # return


        # fig.colorbar(im1, ax=ax[0])
        # fig.colorbar(im2, ax=ax[1])
        # plt.savefig('force.png')
        anim = animation.FuncAnimation(fig, animate, frames=10)
        anim.save(out_path+'/animation.gif',writer='imagemagick', fps=10)