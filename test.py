import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
jobname = sys.argv[1]
# sample_freq = 1
data = pd.read_csv('job/'+jobname+'/src/output/power.csv',header=None).to_numpy()
print(data.shape)
# print(data)
# power = np.reshape(data[0:3*length],[length,3])
# t = np.asarray(range(data[0:-2:3].size))*0.02
# # plt.plot(power[:,0])
# # plt.plot(power[:,1])
# # plt.plot(power[:,2])

plt.figure()
plt.plot(data[0::3,0]/1e6)
plt.plot(data[1::3,0]/1e6)
plt.plot(data[2::3,0]/1e6)

power_1 = data[0::3,0]
power_2 = data[1::3,0]
power_3 = data[2::3,0]
plt.xlim(0,50000)
plt.ylim(0,5)
plt.savefig('test.png')
mean_power_1 = np.mean(power_1[7000:])
mean_power_2 = np.mean(power_2[7000:])
mean_power_3 = np.mean(power_3[7000:])
print(mean_power_1,mean_power_2,mean_power_3)
# plt.show()

# plt.figure()
# jobname = 'NREL-m-superlong-positive'
# data2 = pd.read_csv('job/'+jobname+'/src/output/root_moment.csv',header=None).to_numpy()
# plt.plot(data2[1::2,0]/1e3)
# jobname = 'NREL-m-superlong-negative'
# data2 = pd.read_csv('job/'+jobname+'/src/output/root_moment.csv',header=None).to_numpy()
# plt.plot(data2[1::2,0]/1e3)
# jobname = 'NREL-m-superlong'
# data2 = pd.read_csv('job/'+jobname+'/src/output/root_moment.csv',header=None).to_numpy()
# plt.plot(data2[1::2,0]/1e3)
# # plt.xlim([10000,20000])
# # plt.plot(data2[1:-1:2])
# plt.savefig('test-moment.png')
# # # plt.show()