import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ting_power = [1.0000480261262124,0.5156605193225114,0.5467910543335573,0.53826641693081,0.5435348829763389,0.5462171421253158,0.542002049114718,0.5420981013671431,0.5465044984471547,0.538840329139051]
# plt.imshow(power)

total_power = np.zeros(6)

power = pd.read_csv('../../job/HR1-m-0/src/output/ta_power.dat',header=None).to_numpy()
power = np.reshape(power,[10,8])

total_power[0] = np.sum(power)
print(total_power/1e6)

mean_row_power = np.mean(power,axis=1)
baseline = mean_row_power[0]

fig = plt.figure(dpi=300)
plt.plot(mean_row_power/baseline,'o-',fillstyle='none',label='baseline')

# plt.plot(ting_power,'ko-',fillstyle='none',label='Wu and Porté-Agel (2013)')

# power = pd.read_csv('../../job/HR1-m-5/src/output/ta_power.dat',header=None).to_numpy()
# power = np.reshape(power,[10,8])
# # mean_row_power = np.mean(power,axis=1)
# plt.plot(mean_row_power/baseline,'o-',fillstyle='none',label='$\gamma=5^\circ$')
for i in range(5):
    power = pd.read_csv('../../job/HR1-m-'+str((i+2)*5)+'/src/output/ta_power.dat',header=None).to_numpy()
    power = np.reshape(power,[10,8])
    mean_row_power = np.mean(power,axis=1)
    plt.plot(mean_row_power/baseline,'o-',fillstyle='none',label='$\gamma='+str((i+2)*5)+'^\circ$')
    total_power[i+1] = np.sum(power)
    print(total_power/1e6)
plt.legend()
plt.savefig('power.png')

fig = plt.figure(dpi=300)
plt.plot([0,10,15,20,25,30],total_power/total_power[0]-1,'o-')
plt.ylabel('Normalised total power gains')
plt.xlabel('Yaw angle ($\circ$)')
plt.savefig('total_power.png')



