import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = './job/HR1-m/src/output/ta_power.dat'

power = np.loadtxt(file_path)

# num = 30*10
# df = pd.read_csv(file_path,sep='        ',header=0).to_numpy()
# df =df[200:,:]
print(power.shape)
power = np.reshape(power,[10,8])
# po
# power = power[]
# df =df[100:,:,:]
# print(df.shape)
mean_power = np.mean(power,axis=1)
plt.plot(mean_power/mean_power[0])
# count = np.arange((1390))+1
# result = np.zeros(10)
# for i in range(10):
# plt.plot(mean_power)

#     result[i] = np.cumsum(df[:,i,7])[-1]

# plt.plot(result)
# # for i in range(10):

# # for j in range(10):
# # row = 3
# # plt.plot((np.cumsum(df[:,1,row],axis=0)/np.cumsum(df[:,0,row],axis=0)))
# # plt.plot((np.cumsum(df[:,2,row],axis=0)/np.cumsum(df[:,0,row],axis=0)))
# # plt.plot(np.cumsum(df[:,0],axis=0)/count)
# # # plt.plot(np.cumsum(df[:,2],axis=0)/np.cumsum(df[:,0],axis=0))
# # # plt.plot(np.cumsum(df[:,1],axis=0))
plt.show()
# # print(power.shape)
# # power = np.reshape(power,[10,8])
# print(mean_power)
# # mean_power = np.mean(power,axis=1)
# # mean_power = mean_power/mean_power[0]
# # plt.plot(mean_power)
# # print(mean_power)
# # # # # print(data.shape)
# # # # plt.imshow(data.T)
# # plt.show()

# # # HR_coord = pd.read_csv('./job/HR1-m/input/turb_loc.dat').to_numpy()
# # # print(HR_coord.shape)

# # # plt.figure()
# # # for i in range(HR_coord.shape[0]):
# # #     color_value = power[i]/np.max(power)
# # #     # point1 = [HR_coord[i,0]-2*np.sin(HR_coord[i,3]),HR_coord[i,1]+2*np.cos(HR_coord[i,3])]
# # #     # point2 = [HR_coord[i,0]+2*np.sin(HR_coord[i,3]),HR_coord[i,1]-2*np.cos(HR_coord[i,3])]
# # #     # print(point1
# # #     # plt.plot([point1[0],point2[0]],[point1[1],point2[1]],'k',linewidth=1)
# # #     plt.plot(HR_coord[i,0],HR_coord[i,1],'o',color=((color_value,0,1)))
# # # # plt.xlim([-60,60])
# # # # plt.ylim([-60,60])
# # # # plt.axis('scaled')

# # # # plot_turbine(HR_coord,power)
# # # plt.show()
