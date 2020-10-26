import numpy as np
import pandas as pd 

def plot_turbine(HR_coord,power):
    for i in range(HR_coord.shape[0]):
        color_value = power[i,:]/np.max(power)
        point1 = [HR_coord[i,0]-1.5*np.sin(HR_coord[i,2]),HR_coord[i,1]+1.5*np.cos(HR_coord[i,2])]
        point2 = [HR_coord[i,0]+1.5*np.sin(HR_coord[i,2]),HR_coord[i,1]-1.5*np.cos(HR_coord[i,2])]
        # plt.plot(HR_coord[i,0],HR_coord[i,1],'ko')
        plt.plot([point1[0],point2[0]],[point1[1],point2[1]],color=((color_value,0,1)),linewidth=4)

x_coord = np.linspace(0,63,10)
y_coord = np.zeros(10)
HR_coord = np.zeros([2,10,8])

for i in range(8):
    HR_coord[0,:,i] = x_coord-7*np.sin(np.pi/180*7)*i
    HR_coord[1,:,i] = y_coord+7*np.cos(np.pi/180*7)*i

HR_coord = np.reshape(HR_coord,[2,80]).T
z_coord = np.zeros([80,1])+70
yaw = np.zeros([80,1])
power = np.random.rand(80,1)
HR_coord = np.append(HR_coord,z_coord,axis=1)
HR_coord = np.append(HR_coord,yaw,axis=1)
# plot_turbine(HR_coord,power)
# plt.show()

D = 80
Displacement = np.tile([2560,2560], (80, 1))
HR_coord[:,0:2] = HR_coord[:,0:2]*D+Displacement

df = pd.DataFrame(HR_coord)
df.columns = ["x", "y","z", "gamma"]
df.to_csv("./HornsRev.dat",index=None)
print(HR_coord)