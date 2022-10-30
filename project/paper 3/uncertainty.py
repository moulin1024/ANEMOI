import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

case = ['120s','180s','240s','360s','baseline']
power_baseline = np.zeros([5,8])
for i in range(5):
    print(i)
    casename = 'dyn-yaw-8wt-9-5D-'+case[i]+'-1'
    filepath = '../job/'+casename+'/src/output/ta_power.dat'
    # print(filepath)
    # print(casename)
    power_baseline[i,:] = pd.read_csv(filepath,header=None).to_numpy()[:,0]

# plt.plot(power.T,'.')
total_power = np.sum(power_baseline,axis=1)
print(total_power/total_power[-1]-1)

total_power = np.sum(power_baseline,axis=1)
plt.plot(np.asarray(range(8))+1,power_baseline.T/1e6,'o-')
plt.ylabel('power (mw)')
plt.xlabel('turbine no.')
# plt.hist(total_power)
plt.savefig('uncertainty-9-120s.png')