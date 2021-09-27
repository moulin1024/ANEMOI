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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        dset = grp.create_dataset("uv_std", data=result_3d['uv_std_c'])

        x_grid_unmask = space['x']
        y_grid_unmask = space['y']
        z_grid_unmask = space['z_c']

        x = x_grid_unmask[:]
        y = y_grid_unmask[:]
        z = z_grid_unmask[:-1]

        dset = grp.create_dataset("x", data=x)
        dset = grp.create_dataset("y", data=y)
        dset = grp.create_dataset("z", data=z)

        for key, value in config.items():
            grp.attrs[key]=value

        f.close()
    #     if config['turb_flag'] > 0:
    #         df = pd.read_csv(in_path+'/turb_loc.dat')
    #         df_power = pd.read_csv(src_out_path+'/ta_power.dat',header=None)
    #         df['power'] = df_power
    #         print(df['power'])
    #         print(np.sum(df_power.to_numpy()))
    #         df.to_csv(out_path+'/ta_power.csv',index=False)

    # hub_k = int(config['turb_z']/config['dz'])+1
    # print(hub_k)

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

    # result_4d = post.get_result_4d(src_out_path, config)

    # u = result_4d['u_inst_c']
    # v = result_4d['v_inst_c']
    # w = result_4d['w_inst_c']

    # u2 = u - np.mean(u,axis=0)
    # v2 = v*v - np.mean(v,axis=0)*np.mean(v,axis=0)
    # w2 = w*w - np.mean(w,axis=0)*np.mean(w,axis=0)
    
    # uv = u*v - np.mean(u,axis=0)*np.mean(v,axis=0)
    # vw = v*w - np.mean(v,axis=0)*np.mean(w,axis=0)
    # uw = u*w - np.mean(u,axis=0)*np.mean(w,axis=0)

    # # fig, ax = plt.subplots(1,1)
    # # v_comp = v[9,32,:,:]
    # # w_comp = w[9,32,:,:]
    # # # plt.quiver(v_comp,w_comp)
    # # # plt.ylim([100,160])
    # # # plt.xlim([0,40])
    # mean_plot = np.mean(uv,axis=0)
    # plt.imshow(mean_plot[70,:,:].T,origin='lower',aspect=config['dy']/config['dz'])
    # plt.imshow(mean_plot[:,:,hub_k].T,origin='lower',aspect=config['dy']/config['dz'])
    # # # ax[0].set_xlabel('x')
    # # # ax[0].set_ylabel('y')

    if config['turb_flag'] > 0:
        turb_force = post.get_turb(src_out_path, config)
        fx = turb_force['fx']
        ft = turb_force['ft']
        np.save('fx.npy',fx)
        np.save('ft.npy',ft)
        fx_tot = np.sum(np.sum(np.sum(fx,axis=-1),axis=-1),axis=-1)
        print(fx_tot.shape)
        plt.plot(fx_tot/1000)
        plt.savefig('test.png')
        

    if config['ts_flag'] > 0:
        result_4d = post.get_result_4d(src_out_path, config)
        u = result_4d['u_inst_c']
        v = result_4d['v_inst_c']
        w = result_4d['w_inst_c']

        # np.save('u.npy',u)
        # np.save('v.npy',v)
        # np.save('w.npy',w)

        x_grid_unmask = space['x']
        y_grid_unmask = space['y']
        z_grid_unmask = space['z_c']

        x = x_grid_unmask[config['ts_istart']-1:config['ts_iend']]
        y = y_grid_unmask[config['ts_jstart']-1:config['ts_jend']]
        z = z_grid_unmask[:config['ts_kend']-1]

        print(u.shape)


        if config['turb_flag'] > 0:
            aerodyn = pd.read_csv(in_path+'/turb_aerodyn_nrel.csv',header=None)
            print(aerodyn.shape)
            plt.figure()
            # plt.plot(aerodyn.iloc[:,8])
            # plt.plot(aerodyn.iloc[:,6])
            # plt.plot(aerodyn.iloc[:,4])
            # plt.plot(aerodyn.iloc[:,2])
            plt.plot(aerodyn.iloc[:,0])
            # plt.plot(aerodyn.iloc[:,9])
            plt.plot(aerodyn.iloc[:,10])
            # plt.plot(aerodyn.iloc[:,11])
            plt.savefig('CLCD.png')
            f = h5py.File(out_path+'/'+case_name+'_turb.h5', 'w')
            grp = f.create_group("data")

            dset = grp.create_dataset("fx", data=turb_force['fx'])
            dset = grp.create_dataset("ft", data=turb_force['ft'])

            f.close()

            fx = turb_force['fx']
            plt.figure()
            plt.plot(np.mean(fx[:,0,:,0],axis=0))
            plt.savefig('fx.png')

            ft = turb_force['ft']
            plt.figure()
            plt.plot(np.mean(ft[:,0,:,0],axis=0))
            plt.savefig('ft.png')

            fx_2D = fx[:,0,:,0]
            ft_2D = ft[:,0,:,0]

            np.savetxt("unit_test/elastic-alm/df_fx.csv",fx_2D, delimiter=",")
            np.savetxt("unit_test/elastic-alm/df_ft.csv",ft_2D, delimiter=",")

            time_data = np.asarray(range(ft_2D.shape[0]))*config['dtr']
            np.savetxt("unit_test/elastic-alm/df_time.csv", time_data.T, delimiter=",")

            dr = 63/64
            radius_array = np.zeros(64)

            for i in range(64):
                radius_array[i] = (i+0.5)*dr
            # print(radius_array)
            np.savetxt("unit_test/elastic-alm/df_radius.csv", radius_array.T, delimiter=",")
            
            
        fig,ax = plt.subplots(1,1)
        # values = u[9,:,64,:]
        # plt.imshow(values.T,origin='lower',aspect=config['dz']/config['dy'])
        # plt.colorbar()
        # plt.savefig('test.png')

        def animate(i):    #     azimuths = np.radians(np.linspace(0, 360, 40))
        #     zeniths = np.linspace(0, 0.5, 30)
        #     theta,r = np.meshgrid(azimuths,zeniths,indexing='ij')
            # i = 19
            values = u[i,:,:,30]#np.random.random((azimuths.size, zeniths.size))
            plt.cla()
            im1 = ax.imshow(values.T,origin='lower',aspect=config['dy']/config['dx'])
            # 
            # fig.colorbar(im1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            # values = u[i,32,:,:]#np.random.random((azimuths.size, zeniths.size))
            # im2 = ax[1].imshow(values.T,origin='lower',aspect=config['dz']/config['dx'])
                
        # #     # im2 = ax[1].quiver(v[i,300,1::4,1::4].T,w[i,300,1::4,1::4].T,scale=10)
        # #     # ax[1].scatter([63],[19],marker='+',color='r')
        # #     # ax[1].set_xlabel('y')
        # #     # ax[1].set_ylabel('z')
        # #     # ax[1].colorbar()
            
            print(i)

            return


        # # # # fig.colorbar(im1, ax=ax[0])
        # # # # fig.colorbar(im2, ax=ax[1])
        # # # # plt.savefig('force.png')
        anim = animation.FuncAnimation(fig, animate, frames=10)
        anim.save(out_path+'/animation.gif',writer='pillow', fps=10)
        # fig, ax = plt.subplots(2,3,subplot_kw=dict(projection='polar'))
        # azimuths = np.radians(np.linspace(0, 360, 64))
        # zeniths = np.linspace(0, 0.5, 16)
        # theta,r = np.meshgrid(azimuths,zeniths,indexing='ij')
        # def animate(i):    
        #     for k in range(3):
        #         im1 = ax[0,k].contourf(theta,r,fx[i,:,:,k],40)
        #         im2 = ax[1,k].contourf(theta,r,ft[i,:,:,k],40)
                
        #     print(i)

        # anim = animation.FuncAnimation(fig, animate, frames=19)
        # anim.save(out_path+'/animation_turb.gif',writer='imagemagick', fps=10)


        # fig, ax = plt.subplots(2,3)
        # azimuths = np.radians(np.linspace(0, 360, 64))
        # zeniths = np.linspace(0, 0.5, 16)
        # theta,r = np.meshgrid(azimuths,zeniths,indexing='ij')
        # def animate(i):    
        #     for k in range(3):
        #         im1 = ax[0,k].contourf(theta,r,fx[i,:,:,k],40)
        #         im2 = ax[1,k].contourf(theta,r,ft[i,:,:,k],40)
        #     print(i)

        # anim = animation.FuncAnimation(fig, animate, frames=19)
        # anim.save(out_path+'/animation_turb2.gif',writer='imagemagick', fps=10)
        # fig = plt.figure()
        # ax = plt.axes(xlim=(0, 64), ylim=(0, 6000))
        # line, = ax.plot([], [], lw=2)

        # def init():
        #     line.set_data([], [])
        #     return line,
        # # # i = 9
        # def animate(i):    #     azimuths = np.radians(np.linspace(0, 360, 40))
        # #     zeniths = np.linspace(0, 0.5, 30)
        # #     theta,r = np.meshgrid(azimuths,zeniths,indexing='ij')
        #     print(i)
        #     x = np.arange(64)*63/64
        #     y = fx[i,0,:,0]
        #     line.set_data(x,y)
        #     plt.cla()
        #     ax.plot(x,fx[i,0,:,0],'o')
        #     # ax.set_ylim([0,6000])

        #     return line,
 
        # # fig.colorbar(im1, ax=ax[0])
        # # fig.colorbar(im2, ax=ax[1])
        # # plt.savefig('force.png')
        # anim = animation.FuncAnimation(fig, animate, init_func=init,frames=10, interval=20, blit=True)
        # print("write")
        # anim.save('animation_force.gif',writer='imagemagick', fps=10)

        