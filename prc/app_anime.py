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


        fig,ax = plt.subplots(1,1)
        # plt.rcParams['image.cmap']='Greys'
        def animate(i):    #     azimuths = np.radians(np.linspace(0, 360, 40))
            # values = w[i,:,:,4]#np.random.random((azimuths.size, zeniths.size))
            values = w[i,:,128,:]#np.random.random((azimuths.size, zeniths.size))
            plt.cla()
            # im1 = ax.imshow(values.T,origin='lower',aspect=config['dz']/config['dy'])
            im1 = ax.imshow(values.T,origin='lower',aspect=config['dz']/config['dx'])
            # plt.clim(0,10000)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            # plt.xlim([100,150])
            # plt.ylim([0,100])
            print(i)
            return
        anim = animation.FuncAnimation(fig, animate, frames=10)
        anim.save(out_path+'/animation_xz.gif',writer='pillow', fps=20)

    if config['turb_flag'] > 0:
        turb_force = post.get_turb(src_out_path, config)
        turb_fx =  turb_force['fx']
        turb_ft =  turb_force['ft']

        turb_x = 512 - config['dx']
        turb_y = 512 - config['dy']
        turb_z = 90

        downsample = 100
        flap_scale = 1
        edge_scale = 40
        displacement_flap = turb_force['displacement_flap']
        displacement_edge = turb_force['displacement_edge']
        moment_flap = turb_force['moment_flap']
        moment_edge = turb_force['moment_edge']


        omega = 12.1/69*2*np.pi
        phase_angle = 0
        fig,ax = plt.subplots()
        # ax = fig.add_subplot(1,2,1)
        def animate(i):
            ax.cla() # clear the previous image
            # ax[1,0].cla() # clear the previous image
            # ax[0,1].cla() # clear the previous image
            # ax[1,1].cla() # clear the previous image
            blade = np.linspace(1.5,63,32)
            phase_angle = (i+1)*downsample*omega*config['dtr']
            y_coord_1 = turb_y + blade*np.cos(phase_angle) - displacement_edge[(i+1)*downsample,0,:,0]*edge_scale*np.sin(phase_angle)
            z_coord_1 = turb_z + blade*np.sin(phase_angle) + displacement_edge[(i+1)*downsample,0,:,0]*edge_scale*np.cos(phase_angle)
            x_coord_1 = turb_x + displacement_flap[i,0,:,0]*flap_scale
            y_coord_2 = turb_y + blade*np.cos(phase_angle-2*np.pi/3) - displacement_edge[(i+1)*downsample,1,:,0]*edge_scale*np.sin(phase_angle-2*np.pi/3)
            z_coord_2 = turb_z + blade*np.sin(phase_angle-2*np.pi/3) + displacement_edge[(i+1)*downsample,1,:,0]*edge_scale*np.cos(phase_angle-2*np.pi/3)
            x_coord_2 = turb_x + displacement_flap[i,1,:,0]*flap_scale
            y_coord_3 = turb_y + blade*np.cos(phase_angle-4*np.pi/3) - displacement_edge[(i+1)*downsample,2,:,0]*edge_scale*np.sin(phase_angle-4*np.pi/3)
            z_coord_3 = turb_z + blade*np.sin(phase_angle-4*np.pi/3) + displacement_edge[(i+1)*downsample,2,:,0]*edge_scale*np.cos(phase_angle-4*np.pi/3)
            x_coord_3 = turb_x + displacement_flap[i,2,:,0]*flap_scale

            values = u[i,128,:,:]
            ax.plot(y_coord_1,z_coord_1,'r')
            ax.plot(y_coord_2,z_coord_2,'r')
            ax.plot(y_coord_3,z_coord_3,'r')
            ax.contourf(YY2,ZZ2,values.T,100)
            ax.axis('scaled')
            ax.set_ylim([0,200]) # fix the x axis
            ax.set_xlim([turb_y-200,turb_y+200]) # fix the y axis
            
            print(i)

        anim = animation.FuncAnimation(fig, animate, frames=10)
        anim.save(out_path+'/blade_movement_yz.gif',writer='pillow', fps=12)

        fig,ax = plt.subplots()
        def animate(i):
            ax.cla() # clear the previous image
            # ax[1,0].cla() # clear the previous image
            # ax[0,1].cla() # clear the previous image
            # ax[1,1].cla() # clear the previous image
            blade = np.linspace(1.5,63,32)
            phase_angle = (i+1)*downsample*omega*config['dtr']
            y_coord_1 = turb_y + blade*np.cos(phase_angle) - displacement_edge[(i+1)*downsample,0,:,0]*edge_scale*np.sin(phase_angle)
            z_coord_1 = turb_z + blade*np.sin(phase_angle) + displacement_edge[(i+1)*downsample,0,:,0]*edge_scale*np.cos(phase_angle)
            x_coord_1 = turb_x + displacement_flap[i,0,:,0]*flap_scale
            y_coord_2 = turb_y + blade*np.cos(phase_angle-2*np.pi/3) - displacement_edge[(i+1)*downsample,1,:,0]*edge_scale*np.sin(phase_angle-2*np.pi/3)
            z_coord_2 = turb_z + blade*np.sin(phase_angle-2*np.pi/3) + displacement_edge[(i+1)*downsample,1,:,0]*edge_scale*np.cos(phase_angle-2*np.pi/3)
            x_coord_2 = turb_x + displacement_flap[i,1,:,0]*flap_scale
            y_coord_3 = turb_y + blade*np.cos(phase_angle-4*np.pi/3) - displacement_edge[(i+1)*downsample,2,:,0]*edge_scale*np.sin(phase_angle-4*np.pi/3)
            z_coord_3 = turb_z + blade*np.sin(phase_angle-4*np.pi/3) + displacement_edge[(i+1)*downsample,2,:,0]*edge_scale*np.cos(phase_angle-4*np.pi/3)
            x_coord_3 = turb_x + displacement_flap[i,2,:,0]*flap_scale

            values = w[i,:,128,:]
            ax.plot(x_coord_1,z_coord_1,'r')
            ax.plot(x_coord_2,z_coord_2,'r')
            ax.plot(x_coord_3,z_coord_3,'r')
            ax.contourf(XX3,ZZ3,values.T,100)
            ax.axis('scaled')
            ax.set_ylim([0,200]) # fix the x axis
            ax.set_xlim([turb_y-200,turb_y+200]) # fix the y axis
            
            print(i)

        anim = animation.FuncAnimation(fig, animate, frames=10)
        anim.save(out_path+'/blade_movement.gif',writer='pillow', fps=12)