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
import matplotlib.transforms as transforms
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
# import fatigue
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

    if config['ta_flag'] > 0:
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

        mean_u = np.mean(np.mean(u_avg,axis=0),axis=0)
        print(mean_u[6])
        u_std = result_3d['u_std_c']
        
        mean_std = np.mean(np.mean(u_std,axis=0),axis=0)
        
        print(mean_std[6]/mean_u[6])

    #     fig = figure(figsize=(8,8))
    #     ax = fig.add_subplot(111)
    #     im = ax.imshow(u_avg[:,:,9].T,origin='lower',aspect=0.25)
    #     plt.colorbar(im)
    # #     # print(np.mean(u_avg[176,63-8:63+8,45-32:45+32]))
    # #     # im = ax.imshow()
    # #     im = ax.quiver(v_avg[176,::4,::4].T,w_avg[176,::4,::4].T,scale=50)
    #     plt.savefig(out_path+'/test.png')
    #     print('check')



    if config['ts_flag'] > 0:

        # u  = fctlib.load_3d('001_ts_u', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)
    
        f = h5py.File(out_path+'/'+case_name+'_flowfield.h5','w')
        # for key, value in config.items():
        #     f.attrs[key] = value
        # result_4d = post.get_result_4d(src_out_path, config)
        
        # f.create_dataset('x',data=space['x'])
        # f.create_dataset('y',data=space['y'])
        # f.create_dataset('z',data=space['z_c'])

        # f.create_dataset('t_sample',data=time['t_ts'])
        # f.create_dataset('u',data=result_4d['u_inst_c'][:,:,:])
        # f.create_dataset('v',data=result_4d['v_inst_c'])
        # f.create_dataset('w',data=result_4d['w_inst_c'])
        # print(space['y'].shape)
        # f.close
        t_count = (config['nsteps']-config['ts_tstart'])//100
        # q_data = np.zeros([t_count,config['nx'],config['ny'],config['nz']-1])
        velo_data = np.zeros([t_count,config['nx'],config['ny'],3])
        for i in range(t_count):
            # print(i)
            # qcrit = fctlib.load_3d(str(i).zfill(4)+'_ts_slice_u', config['nx'],  config['ny'], config['double_flag'], src_out_path)
            u = fctlib.load_3d(str(i).zfill(3)+'_ts_u', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)[:,:,:-1]
            v = fctlib.load_3d(str(i).zfill(3)+'_ts_v', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)[:,:,:-1]
            w = fctlib.load_3d(str(i).zfill(3)+'_ts_w', config['nx'],  config['ny'],  config['nz'], config['double_flag'], src_out_path)[:,:,:-1]

            velo_data[i,:,:,0] = np.flip(u[:,:,int(config['lz']/config['nz'])],axis=0)
            velo_data[i,:,:,1] = np.flip(v[:,:,int(config['lz']/config['nz'])],axis=0)
            velo_data[i,:,:,2] = np.flip(w[:,:,int(config['lz']/config['nz'])],axis=0)
            
            print(i)

            fig = figure(figsize=(8,6),dpi=100)
            # ax1 = fig.add_subplot(111)
            plt.imshow(u[128,:,:].T,origin='lower',aspect=1/2,vmin=1,vmax=11)
            plt.colorbar()
            plt.savefig(out_path+'/'+str(i).zfill(3)+'_flowfield_xz.png')
            plt.close()
            # # Define the rotation angle in degrees
            # rotation_degrees = 5

            # data = np.flip(u[:,:,7].T,axis=0)*np.cos(np.deg2rad(-rotation_degrees)) - np.flip(-v[:,:,7].T,axis=0)*np.sin(np.deg2rad(-rotation_degrees))
            
            # # data = np.flip(-v[:,:,7].T,axis=0)*np.cos(np.deg2rad(rotation_degrees)) + np.flip(u[:,:,7].T,axis=0)*np.sin(np.deg2rad(rotation_degrees))
            

            # fig, ax = plt.subplots()
            # # plt.rcParams["font.size"] = "16"
            # im = ax.imshow(data,origin='lower',extent=[-5120,config['lx']-5120,-2560,config['ly']-2560],vmin=2,vmax=10,cmap='bwr')


            # # Define the center of rotation
            # rotation_center = (10240 / 2, 5120 / 2)

            # # Create a translation transformation to move the center of rotation to the origin
            # translation_to_origin = transforms.Affine2D().translate(-rotation_center[0], -rotation_center[1])

            # # Create a rotation transformation
            # rotation = transforms.Affine2D().rotate_deg(rotation_degrees)

            # # Create a translation transformation to move the center of rotation back to its original position
            # translation_to_center = transforms.Affine2D().translate(rotation_center[0], rotation_center[1])

            # # Combine the transformations
            # combined_transform = translation_to_origin + rotation + translation_to_center

            # # Apply the combined transformation to the image
            # im.set_transform(combined_transform + ax.transData)

            # # plt.colorbar()
            # ax.set_xlabel('x (m)')
            # ax.set_ylabel('z (m)')
            # ax.set_xlim([1500-5120,9000-5120])
            # ax.set_ylim([320-2560,5120-320-2560])
            # cbar = plt.colorbar(im,ax=ax)
            # cbar.set_label('$u$ (m/s)')
            # plt.savefig(out_path+'/'+str(i).zfill(3)+'_flowfield_yz.png',bbox_inches='tight')
            # plt.close()

        # f.create_dataset('q_criterion',data=q_data)
        # f.create_dataset('hub_height_velocity',data=velo_data)
        
        # f.close
        

    if config['turb_flag'] > 0:
        turb_loc = pd.read_csv(case_path+"/input/turb_loc.dat")
        f = h5py.File(out_path+'/'+case_name+'_force.h5','w')
        for key, value in config.items():
            f.attrs[key] = value
        # f.create_dataset('turb_x',data=turb_loc['x'].to_numpy())
        # f.create_dataset('turb_y',data=turb_loc['y'].to_numpy())
        # f.create_dataset('turb_z',data=turb_loc['z'].to_numpy())
        # f.create_dataset('yaw',data=turb_loc['yaw'].to_numpy())
        # f.create_dataset('tilt',data=turb_loc['yaw'].to_numpy())
        
        # print(turb_loc['yaw'].to_numpy())
        turb_force = post.get_turb(src_out_path, config)
        # power = np.squeeze(np.squeeze(turb_force['power'], axis=1), axis=1)
        # print(power.shape)
        # plt.plot(power[:,0])
        # plt.savefig('test.png')
        f.create_dataset('time',data=time['t'])
        f.create_dataset('fx',data=turb_force['fx'][:,:,:,:])
        f.create_dataset('ft',data=turb_force['ft'][:,:,:,:])
        # f.create_dataset('displacement_flap',data=turb_force['displacement_flap'])
        # f.create_dataset('displacement_edge',data=turb_force['displacement_edge'])
        f.create_dataset('moment_flap',data=turb_force['moment_flap'])
        f.create_dataset('moment_edge',data=turb_force['moment_edge'])
        # f.create_dataset('velocity_flap',data=turb_force['velocity_flap'])
        # f.create_dataset('velocity_edge',data=turb_force['velocity_edge'])
        f.create_dataset('phase',data=turb_force['phase'])
        f.create_dataset('inflow',data=turb_force['inflow'])

        f.close
        # print(turb_force['ft'].shape)
        plt.figure()
        print(turb_force['inflow'].shape)
        plt.plot(turb_force['inflow'][:,0,0,0])

        # plt.plot(turb_force['moment_flap'][:,0,0,0])
        # plt.plot(turb_force['moment_edge'][:,0,0,0])
        plt.savefig('test.png')
        # displacement_flap = turb_force['displacement_flap']
        # displacement_edge = turb_force['displacement_edge']

        # print(displacement_flap.shape)
        # plt.plot(displacement_flap[:,0,-1,0])
        # # print(displacement_flap[:,0,0,0])
        # # plt.xlim([900,1000])
        # plt.savefig('test.png')
        # dt = config['dt']
        # downsample = 10
        # flap_scale = 1
        # edge_scale = 10
        
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
        #     ax4.set_xlim([512-128,512+128])
        #     ax4.set_ylim([512-128,512+128])
        #     ax4.set_zlim([0,256])
        #     # ax4.set_box_aspect([1,1,1])
        #     # ax4.azim = -90
        #     # ax4.elev = 0

        #     print(i)

        # anim = animation.FuncAnimation(fig, animate, frames=1)
        # anim.save(out_path+'/blade_movement_3d.gif',writer='pillow', fps=20)
