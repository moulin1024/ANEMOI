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

from matplotlib import projections
import fctlib
import app_post as post
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import os.path
import math
import pandas as pd
from pathlib import Path
from matplotlib.pyplot import figure
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D
import fatigue
# from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import cv2
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

    if config['ta_flag'] > 20:
        result_3d = post.get_result_3d(src_inp_path,src_out_path, config)
        u_avg = result_3d['u_avg_c']
        v_avg = result_3d['v_avg_c']
        w_avg = result_3d['w_avg_c']

        u_std = result_3d['u_std_c']
        v_std = result_3d['v_std_c']
        w_std = result_3d['w_std_c']

        f = h5py.File(out_path+'/'+case_name+'_stat.h5','w')
        for key, value in config.items():
            f.attrs[key] = value

        f.create_dataset('x',data=space['x'])
        f.create_dataset('y',data=space['y'])
        f.create_dataset('z',data=space['z_c'])

        f.create_dataset('u_avg',data=u_avg )
        f.create_dataset('v_avg',data=v_avg)
        f.create_dataset('w_avg',data=w_avg)

        f.create_dataset('u_std',data=u_std)
        f.create_dataset('v_std',data=v_std)
        f.create_dataset('w_std',data=w_std)

        f.close

        fig = figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        im = ax.imshow(u_avg[:,:,9].T,origin='lower',aspect=0.25)
        plt.colorbar(im)
    #     # print(np.mean(u_avg[176,63-8:63+8,45-32:45+32]))
    #     # im = ax.imshow()
    #     im = ax.quiver(v_avg[176,::4,::4].T,w_avg[176,::4,::4].T,scale=50)
        plt.savefig(out_path+'/test.png')
    #     print('check')



    if config['ts_flag'] > 10:

        # u  = fctlib.load_3d('001_ts_u', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)
        mean_induction = np.zeros([511,128])
        f = h5py.File(out_path+'/'+case_name+'_flowfield.h5','w')
        # for key, value in config.items():
        #     f.attrs[key] = value
        # result_4d = post.get_result_4d(src_out_path, config)
        
        f.create_dataset('x',data=space['x'])
        f.create_dataset('y',data=space['y'])
        f.create_dataset('z',data=space['z_c'])
        x = space['x']
        y = space['y'] - 255
        z = space['z_c'] - 90

        yy,zz = np.meshgrid(y,z,indexing='xy')
        phi = np.arctan(zz/yy)
        r = np.sqrt(yy**2 + zz**2)
        f.create_dataset('t_sample',data=time['t_ts'])
        # f.create_dataset('u',data=result_4d['u_inst_c'][:,:,:])
        # f.create_dataset('v',data=result_4d['v_inst_c'])
        # f.create_dataset('w',data=result_4d['w_inst_c'])
        # print(space['y'].shape)
        f.close
        t_count = (config['nsteps']-config['ts_tstart'])//100
        velo_data = np.zeros([t_count,config['nx'],config['ny'],3])

        for i in range(10):
            # print(i)
            # qcrit = fctlib.load_3d(str(i).zfill(4)+'_ts_slice_u', config['nx'],  config['ny'], config['double_flag'], src_out_path)
            # qcrit = fctlib.load_3d(str(i).zfill(3)+'_ts_v', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)[:,:,:-1]
            u = fctlib.load_3d(str(i).zfill(3)+'_ts_u', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)[:,:,:-1]
            v = fctlib.load_3d(str(i).zfill(3)+'_ts_v', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)[:,:,:-1]
            w = post.node2center_3d(fctlib.load_3d(str(i).zfill(3)+'_ts_w', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path))
            
            # velo_data[i,2,:,:,:] = w[128:,64-16:64+16,:128]
            # velo_data[i,:,:] = u[:,:,90]
            # velo_data[i,1,:,:,:] = v[128:,64-16:64+16,:128]
            
            print(i)
            velo_data[i,:,:,0] = u[:,:,90]
            velo_data[i,:,:,1] = v[:,:,90]
            velo_data[i,:,:,2] = w[:,:,90]
            fig = figure(figsize=(8,6),dpi=100)
            ax1 = fig.add_subplot(111)
            induction = (1 - u[64,:,:].T/u[44,:,:].T)
            mean_induction = induction + mean_induction
            tang_v = -v[64,:,:].T*zz/r+w[64,:,:].T*yy/r
            tang_induction = tang_v/(1.71*r)

            # mean_induction = tang_induction + mean_induction
            # im = ax1.imshow(mean_induction/(i+1),vmin=0,vmax=0.3,origin='lower',aspect=1/4)

            im = ax1.imshow(tang_induction,origin='lower',aspect=1/4)
            # im = ax1.contourf(y,z,phi)
            plt.colorbar(im)
            plt.savefig(out_path+'/'+str(i).zfill(3)+'_flowfield_xz.png')
            plt.close()

        f.create_dataset('velocity',data=velo_data)
        f.close
        # u = np.flip(result_4d['u_inst_c'],axis=2)
        # v = - np.flip(result_4d['v_inst_c'],axis=2)
        # w = np.flip(result_4d['w_inst_c'],axis=2)

        # dx = space['x'][1]-space['x'][0]
        # dy = space['y'][1]-space['y'][0]
        # dz = space['z_c'][1]-space['z_c'][0]

        # for i in range(100):
        #     print(i)
        #     fig = figure(figsize=(8,6),dpi=300)
        #     ax1 = fig.add_subplot(111,projection='3d')
        #     u = fctlib.load_3d(str(i+901).zfill(3)+'_ts_u', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)[:,:,:-1]
        #     v = fctlib.load_3d(str(i+901).zfill(3)+'_ts_v', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)[:,:,:-1]
        #     w = post.node2center_3d(fctlib.load_3d(str(i+901).zfill(3)+'_ts_w', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path))
        
        #     dudx = np.diff(u,axis=0)[:,:-1,:-1]/dx
        #     dudy = np.diff(u,axis=1)[:-1,:,:-1]/dy
        #     dudz = np.diff(u,axis=2)[:-1,:-1,:]/dz
        #     dvdx = np.diff(v,axis=0)[:,:-1,:-1]/dx
        #     dvdy = np.diff(v,axis=1)[:-1,:,:-1]/dy
        #     dvdz = np.diff(v,axis=2)[:-1,:-1,:]/dz
        #     dwdx = np.diff(w,axis=0)[:,:-1,:-1]/dx
        #     dwdy = np.diff(w,axis=1)[:-1,:,:-1]/dy
        #     dwdz = np.diff(w,axis=2)[:-1,:-1,:]/dz
        #     Qcriter=dudx*dvdy+dvdy*dwdz+dwdz*dudx-dudy*dvdx-dvdz*dwdy-dwdx*dudz
        #     verts, faces, _, _ = marching_cubes(Qcriter, 0.02)
        #     ax1.plot_trisurf(verts[:, 0]*dx, verts[:,1]*dy, faces, verts[:, 2]*dz, lw=1,alpha=0.2)
        #     ax1.set_xlabel('x')
        #     ax1.set_ylabel('y')
        #     ax1.set_xlim(0,4096)
        #     ax1.set_ylim(0,1024)
        #     ax1.set_zlim(0,512)
        #     ax1.set_box_aspect([8,2,1])
        #     plt.savefig(out_path+'/'+str(i+1).zfill(3)+'_animation_xz.png')
        #     plt.close()


    if config['turb_flag'] > 0:
        turb_loc = pd.read_csv(case_path+"/input/turb_loc.dat")
        f = h5py.File(out_path+'/'+case_name+'_force.h5','w')
        for key, value in config.items():
            f.attrs[key] = value
        f.create_dataset('turb_x',data=turb_loc['x'].to_numpy())
        f.create_dataset('turb_y',data=turb_loc['y'].to_numpy())
        f.create_dataset('turb_z',data=turb_loc['z'].to_numpy())
        f.create_dataset('yaw',data=turb_loc['yaw'].to_numpy())
        f.create_dataset('tilt',data=turb_loc['yaw'].to_numpy())
        
        print(turb_loc['yaw'].to_numpy())
        turb_force = post.get_turb(src_out_path, config)
        # power = np.squeeze(np.squeeze(turb_force['power'], axis=1), axis=1)
        # print(power.shape)
        # plt.plot(power[:,0])
        # plt.savefig('test.png')
        f.create_dataset('time',data=time['t'][::2])
        # f.create_dataset('fx',data=turb_force['fx'][:,:,:,:])
        # f.create_dataset('ft',data=turb_force['ft'][:,:,:,:])
        # f.create_dataset('displacement_flap',data=turb_force['displacement_flap'][:,:,:,:])
        # f.create_dataset('displacement_edge',data=turb_force['displacement_edge'][:,:,:,:])
        f.create_dataset('moment_flap',data=turb_force['moment_flap'][::2,:,:,:])
        f.create_dataset('moment_edge',data=turb_force['moment_edge'][::2,:,:,:])
        # f.create_dataset('velocity_flap',data=turb_force['velocity_flap'])
        # f.create_dataset('velocity_edge',data=turb_force['velocity_edge'])
        f.create_dataset('phase',data=turb_force['phase'][::2,:])

        f.close

        displacement_flap = turb_force['moment_flap']
        displacement_edge = turb_force['moment_edge']

        print(displacement_flap.shape)
        plt.plot(displacement_flap[:,0,0,0])
        print(displacement_flap[:,0,0,0])
        # plt.xlim([900,1000])
        plt.savefig('test.png')
        # dt = config['dt']
        # downsample = 10
        # flap_scale = 1
        # edge_scale = 1
        
        # initial_phase = [0,0,0]
        # blade_coord = np.zeros([32,3,3])
        # blade_coord_tilted = np.zeros([32,3,3])
        # blade_coord_yawed = np.zeros([32,3,3])
        # blade = np.linspace(1.5,63,32)
        # tilt_angle = 0.0#np.deg2rad(df['tilt'][0])
        # print(config['dyn_yaw_freq'])
        # dyn_yaw_freq = config['dyn_yaw_freq'][2:-2].split(",")
        # dyn_yaw_freq = [float(item) for item in dyn_yaw_freq]
        # print(dyn_yaw_freq)
        # fig = figure(figsize=(8,8),dpi=200)
        # # ax1 = fig.add_subplot(111)
        # # ax2 = fig.add_subplot(222)
        # # ax3 = fig.add_subplot(223)
        # ax4 = fig.add_subplot(111,projection='3d')
        # def animate(i):

        # #     ax1.cla() # clear the previous image
        # #     ax2.cla() # clear the previous image
        # #     ax3.cla() # clear the previous image
        #     ax4.cla() # clear the previous image
        #     for i_turb in range(1):
        #         phase = np.squeeze(np.squeeze(turb_force['phase'],axis=3),axis=1)
        #         # phase = np.squeeze(turb_force['phase'][:,:,:,i_turb],axis=1)
        #         turb_x=turb_loc['x'].to_numpy()[i_turb]
        #         turb_y=turb_loc['y'].to_numpy()[i_turb]
        #         turb_z=turb_loc['z'].to_numpy()[i_turb]
        #         # print(turb_x)

        #         # ax4.plot3D([turb_x,turb_x],[turb_y,turb_y],[0,90],'k')
        #         yaw_angle = np.sin((i+1)*(config['dtr']*downsample)*2*np.pi*dyn_yaw_freq[i_turb])*turb_loc['yaw'].to_numpy()[i_turb]/180*np.pi
        #         for j in range(3):
        #             # ax4.plot3D([turb_x+j*100,turb_x+j*100],[turb_y,turb_y],[0,90],'k')

        #             blade_coord[:,j,0] = displacement_flap[(i+1)*downsample,j,:,i_turb]*flap_scale
        #             blade_coord[:,j,1] = blade*np.cos(phase[(i+1)*downsample,j]) - displacement_edge[(i+1)*downsample,j,:,i_turb]*edge_scale*np.sin(phase[(i+1)*downsample,j])
        #             blade_coord[:,j,2] = blade*np.sin(phase[(i+1)*downsample,j]) + displacement_edge[(i+1)*downsample,j,:,i_turb]*edge_scale*np.cos(phase[(i+1)*downsample,j])
                    
        #             blade_coord_tilted[:,j,0] = blade_coord[:,j,0]*np.cos(tilt_angle) + blade_coord[:,j,2]*np.sin(tilt_angle)
        #             blade_coord_tilted[:,j,1] = blade_coord[:,j,1]
        #             blade_coord_tilted[:,j,2] = blade_coord[:,j,2]*np.cos(tilt_angle) - blade_coord[:,j,0]*np.sin(tilt_angle)
                    
        #             blade_coord_yawed[:,j,0] = blade_coord_tilted[:,j,0]*np.cos(yaw_angle) - blade_coord_tilted[:,j,1]*np.sin(yaw_angle)  
        #             blade_coord_yawed[:,j,1] = blade_coord_tilted[:,j,1]*np.cos(yaw_angle) + blade_coord_tilted[:,j,0]*np.sin(yaw_angle) 
        #             blade_coord_yawed[:,j,2] = blade_coord_tilted[:,j,2] 

        #             # print(turb_x)
                    
        #             blade_coord_yawed[:,j,0] = turb_x + blade_coord_yawed[:,j,0]
        #             blade_coord_yawed[:,j,1] = turb_y + blade_coord_yawed[:,j,1]
        #             blade_coord_yawed[:,j,2] = turb_z + blade_coord_yawed[:,j,2]
        #             # print(blade_coord_yawed.shape)
        #     #         ax1.plot(blade_coord_yawed[:,j,1], blade_coord_yawed[:,j,2])
        #     #         ax2.plot(blade_coord_yawed[:,j,0], blade_coord_yawed[:,j,2])
        #     #         ax3.plot(blade_coord_yawed[:,j,0], blade_coord_yawed[:,j,1])
        #             ax4.scatter3D(blade_coord_yawed[:,j,0],blade_coord_yawed[:,j,1], blade_coord_yawed[:,j,2],'.',s=1)

        # #     ax1.axis('scaled')
        # #     ax1.set_xlabel('y (m)')
        # #     ax1.set_ylabel('z (m)')
        # #     ax1.set_ylim([0,200]) # fix the x axis
        # #     ax1.set_xlim([turb_y-100,turb_y+100]) # fix the y axis
            
        # #     ax2.axis('scaled')
        # #     ax2.set_xlabel('x (m)')
        # #     ax2.set_ylabel('z (m)')
        # #     ax2.set_ylim([0,200]) # fix the x axis
        # #     ax2.set_xlim([turb_x-100,turb_x+100]) # fix the y axis

        # #     ax3.axis('scaled')
        # #     ax3.set_xlabel('x (m)')
        # #     ax3.set_ylabel('y (m)')
        # #     ax3.set_ylim([turb_y-100,turb_y+100]) # fix the x axis
        # #     ax3.set_xlim([turb_x-100,turb_x+100]) # fix the y axis

        #     # ax4.set_xlabel('x (m)')
        #     # ax4.set_ylabel('y (m)')
        #     # ax4.set_zlabel('z (m)')
        #     ax4.set_xlim([1024-128,1024+128])
        #     ax4.set_ylim([512-128,1024+128])
        #     ax4.set_zlim([0,256])
        #     ax4.set_box_aspect([1,1,1])
        #     ax4.azim = -90
        #     ax4.elev = 0

        #     print(i)

        # anim = animation.FuncAnimation(fig, animate, frames=100)
        # anim.save(out_path+'/blade_movement_3d.gif',writer='pillow', fps=20)