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
from mpl_toolkits.mplot3d import Axes3D
import fatigue
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
    print(case_path)

    df = pd.read_csv(case_path+"/input/turb_loc.dat")
    # print(df['yaw'][0])

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

        XX1,YY1 = np.meshgrid(x,y)
        YY2,ZZ2 = np.meshgrid(y,z)
        XX3,ZZ3 = np.meshgrid(x,z)


        # fig,ax = plt.subplots(1,1)
        # def animate(i):    
        #     values = u[i,:,:,45]
        #     plt.cla()
        #     im1 = ax.imshow(values.T,origin='lower',aspect=config['dx']/config['dy'])
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     print(i)
        #     return
        # anim = animation.FuncAnimation(fig, animate, frames=10)
        # anim.save(out_path+'/animation_xz.gif',writer='pillow', fps=10)

    if config['turb_flag'] > 0:
        turb_force = post.get_turb(src_out_path, config)
        turb_fx =  turb_force['fx']
        turb_ft =  turb_force['ft']

        turb_x = 512 - config['dx']
        turb_y = 512 - config['dy']
        turb_z = 90

        downsample = 10
        flap_scale = 1
        edge_scale = 50
        displacement_flap = turb_force['displacement_flap']
        displacement_edge = turb_force['displacement_edge']
        moment_flap = turb_force['moment_flap']
        moment_edge = turb_force['moment_edge']

        print(moment_edge.shape)

        initial_phase = [0,-2*np.pi/3,-4*np.pi/3]
        blade_coord = np.zeros([32,3,3])
        blade_coord_tilted = np.zeros([32,3,3])
        blade_coord_yawed = np.zeros([32,3,3])
        blade = np.linspace(1.5,63,32)
        omega = 12.1/69*2*np.pi
        phase_angle = 0
        tilt_angle = np.deg2rad(df['tilt'][0])
        yaw_angle = np.deg2rad(df['yaw'][0])
        
        # total_thrust = np.sum(np.sum(np.sum(turb_fx,axis=-1),axis=-1),axis=-1)
        # plt.plot(moment_flap[:,0,0,0])
        # plt.plot(moment_flap[:,1,0,0])
        # plt.plot(moment_flap[:,2,0,0])
        # plt.savefig(out_path+'/total force.png')
        
        Neq = 1000
        M_eq_baseline = fatigue.get_DEL(moment_edge[20000:,0,0,0],Neq,10)
        print(M_eq_baseline)
        M_eq_baseline = fatigue.get_DEL(moment_edge[20000:,1,0,0],Neq,10)
        print(M_eq_baseline)
        M_eq_baseline = fatigue.get_DEL(moment_edge[20000:,2,0,0],Neq,10)
        print(M_eq_baseline)

        arr2D = np.array([moment_flap[:,0,0,0],moment_edge[:,0,0,0]]).T
        np.savetxt(out_path + '/root_moment.csv', arr2D, delimiter=',', fmt='%d')
        # # print(turb_ft)


        # fig = figure()
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        # ax3 = fig.add_subplot(223)
        # ax4 = fig.add_subplot(224,projection='3d')

        # def animate(i):

        #     ax1.cla() # clear the previous image
        #     ax2.cla() # clear the previous image
        #     ax3.cla() # clear the previous image
        #     ax4.cla() # clear the previous image
        #     phase_angle = (i+1)*downsample*omega*config['dtr']
        #     for j in range(3):
        #         blade_coord[:,j,0] = displacement_flap[(i+1)*downsample,j,:,0]*flap_scale
        #         blade_coord[:,j,1] = blade*np.cos(phase_angle+initial_phase[j]) - displacement_edge[(i+1)*downsample,j,:,0]*edge_scale*np.sin(phase_angle+initial_phase[j])
        #         blade_coord[:,j,2] = blade*np.sin(phase_angle+initial_phase[j]) + displacement_edge[(i+1)*downsample,j,:,0]*edge_scale*np.cos(phase_angle+initial_phase[j])
                
        #         blade_coord_tilted[:,j,0] = blade_coord[:,j,0]*np.cos(tilt_angle) + blade_coord[:,j,2]*np.sin(tilt_angle)
        #         blade_coord_tilted[:,j,1] = blade_coord[:,j,1]
        #         blade_coord_tilted[:,j,2] = blade_coord[:,j,2]*np.cos(tilt_angle) - blade_coord[:,j,0]*np.sin(tilt_angle)
                
        #         blade_coord_yawed[:,j,0] = blade_coord_tilted[:,j,0]*np.cos(yaw_angle) - blade_coord_tilted[:,j,1]*np.sin(yaw_angle)  
        #         blade_coord_yawed[:,j,1] = blade_coord_tilted[:,j,1]*np.cos(yaw_angle) + blade_coord_tilted[:,j,0]*np.sin(yaw_angle) 
        #         blade_coord_yawed[:,j,2] = blade_coord_tilted[:,j,2] 


        #         blade_coord_yawed[:,j,0] = turb_x + blade_coord_yawed[:,j,0]
        #         blade_coord_yawed[:,j,1] = turb_y + blade_coord_yawed[:,j,1]
        #         blade_coord_yawed[:,j,2] = turb_z + blade_coord_yawed[:,j,2]

        #         ax1.plot(blade_coord_yawed[:,j,1], blade_coord_yawed[:,j,2])
        #         ax2.plot(blade_coord_yawed[:,j,0], blade_coord_yawed[:,j,2])
        #         ax3.plot(blade_coord_yawed[:,j,0], blade_coord_yawed[:,j,1])
        #         ax4.plot3D(blade_coord_yawed[:,j,0],blade_coord_yawed[:,j,1], blade_coord_yawed[:,j,2])

        #     ax1.axis('scaled')
        #     ax1.set_xlabel('y (m)')
        #     ax1.set_ylabel('z (m)')
        #     ax1.set_ylim([0,200]) # fix the x axis
        #     ax1.set_xlim([turb_y-100,turb_y+100]) # fix the y axis
            
        #     ax2.axis('scaled')
        #     ax2.set_xlabel('x (m)')
        #     ax2.set_ylabel('z (m)')
        #     ax2.set_ylim([0,200]) # fix the x axis
        #     ax2.set_xlim([turb_x-100,turb_x+100]) # fix the y axis

        #     ax3.axis('scaled')
        #     ax3.set_xlabel('x (m)')
        #     ax3.set_ylabel('z (m)')
        #     ax3.set_ylim([turb_y-100,turb_y+100]) # fix the x axis
        #     ax3.set_xlim([turb_x-100,turb_x+100]) # fix the y axis

        #     ax4.set_xlabel('x (m)')
        #     ax4.set_ylabel('y (m)')
        #     ax4.set_zlabel('z (m)')
        #     ax4.set_zlim([0,200]) # fix the x axis
        #     ax4.set_xlim([turb_x-100,turb_x+100]) # fix the y axis
        #     ax4.set_ylim([turb_y-100,turb_y+100]) # fix the y axis

        #     print(i)

        # anim = animation.FuncAnimation(fig, animate, frames=10)
        # anim.save(out_path+'/blade_movement_yz.gif',writer='pillow', fps=12)
