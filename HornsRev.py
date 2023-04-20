import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sys

inflow_angle = np.deg2rad(np.float(sys.argv[1]))
print(inflow_angle)
# def plot_turbine(HR_coord,power):
#     for i in range(HR_coord.shape[0]):
#         color_value = power[i,:]/np.max(power)
#         point1 = [HR_coord[i,0]-1.5*np.sin(HR_coord[i,2]),HR_coord[i,1]+1.5*np.cos(HR_coord[i,2])]
#         point2 = [HR_coord[i,0]+1.5*np.sin(HR_coord[i,2]),HR_coord[i,1]-1.5*np.cos(HR_coord[i,2])]
#         # plt.plot(HR_coord[i,0],HR_coord[i,1],'ko')
#         plt.plot([point1[0],point2[0]],[point1[1],point2[1]],color=((color_value,0,1)),linewidth=4)

x_coord = np.linspace(0,63,10)
y_coord = np.zeros(10)
HR_coord = np.zeros([2,10,8])

for i in range(8):
    HR_coord[0,:,i] = x_coord-7*np.sin(np.pi/180*7)*i
    HR_coord[1,:,i] = y_coord+7*np.cos(np.pi/180*7)*i


HR_coord = np.reshape(HR_coord,[2,80]).T
HR_coord[:,0] = HR_coord[:,0] - 4.5*7
HR_coord[:,1] = HR_coord[:,1] - 3.5*7

HR_coord_rotate_x = HR_coord[:,0]*np.cos(inflow_angle) + HR_coord[:,1]*np.sin(inflow_angle)
HR_coord_rotate_y = - HR_coord[:,0]*np.sin(inflow_angle) + HR_coord[:,1]*np.cos(inflow_angle)

HR_coord[:,0] = HR_coord_rotate_x
HR_coord[:,1] = HR_coord_rotate_y

z_coord = np.zeros([80,1])+70
yaw = np.zeros([80,1])
tilt = np.zeros([80,1])
# yaw_2d = np.reshape(yaw,[10,8])
# yaw_2d[-1,:] = np.pi/6
# yaw = np.reshape(yaw_2d,[80,1])
power = np.random.rand(80,1)
HR_coord = np.append(HR_coord,z_coord,axis=1)
HR_coord = np.append(HR_coord,yaw,axis=1)
HR_coord = np.append(HR_coord,tilt,axis=1)
# print(HR_coord[0,3])

plt.figure()
for i in range(HR_coord.shape[0]):
    color_value = power[i,:]/np.max(power)
    point1 = [HR_coord[i,0]-2*np.sin(HR_coord[i,3]),HR_coord[i,1]+2*np.cos(HR_coord[i,3])]
    point2 = [HR_coord[i,0]+2*np.sin(HR_coord[i,3]),HR_coord[i,1]-2*np.cos(HR_coord[i,3])]
    print(point1)
    plt.plot([point1[0],point2[0]],[point1[1],point2[1]],'k',linewidth=1)
    plt.plot(HR_coord[i,0],HR_coord[i,1],'o',color=((color_value[0],0,1)))
# plt.xlim([-60,60])
# plt.ylim([-60,60])
# plt.axis('scaled')

# plot_turbine(HR_coord,power)
plt.show()

# 4.5*560
# 3.5*560
D = 80
Displacement = np.tile([6120,2560], (80, 1))
HR_coord[:,0:2] = HR_coord[:,0:2]*D+Displacement
HR_coord[:,3] = HR_coord[:,3]/np.pi*180 
df = pd.DataFrame(HR_coord)
df.columns = ["x", "y","z", "gamma","tilt"]
df.to_csv("./HornsRev.dat",index=None)
print(HR_coord)