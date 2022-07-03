import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

jobname = 'NREL-m'
sample_freq = 1
data = pd.read_csv('job/'+jobname+'/src/output/power.csv',header=None).to_numpy()
print(data.shape[0])
# power = np.reshape(data[0:3*length],[length,3])
t = np.asarray(range(data.shape[0]))*0.02
# plt.plot(power[:,0])
# plt.plot(power[:,1])
# plt.plot(power[:,2])
plt.figure()
plt.plot(t[0:-1],data[0:-1]/1e6)
plt.xlim(50,200)
plt.ylim(2,6)
# plt.savefig('test.png')
plt.show()

# data2 = pd.read_csv('job/'+jobname+'/src/output/root_moment.csv',header=None).to_numpy()
# plt.figure()
# plt.plot(data2[0:-1:2])
# plt.plot(data2[1:-1:2])
# # plt.savefig('test.png')
# plt.show()