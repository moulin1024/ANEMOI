import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

baseline_power_7d = pd.read_csv('../../job/7D-baseline/src/output/ta_power.dat',header=None).to_numpy()[:,0]
total_baseline_power_7d = np.sum(baseline_power_7d)

# print(total_baseline_power)
power_7d = np.zeros([8,8])
for i in range(8):
    # for j in range(9):
    power_7d[:,i] = pd.read_csv('../../job/7D-freq'+str((i+1)*2)+'e3-yaw10/src/output/ta_power.dat',header=None).to_numpy()[:,0]

baseline_power_5d = pd.read_csv('../../job/5D-baseline/src/output/ta_power.dat',header=None).to_numpy()[:,0]
total_baseline_power_5d = np.sum(baseline_power_5d)

# print(total_baseline_power)
power_5d = np.zeros([8,8])
for i in range(8):
    # for j in range(9):
    power_5d[:,i] = pd.read_csv('../../job/5D-freq'+str((i+1)*2)+'e3-yaw10/src/output/ta_power.dat',header=None).to_numpy()[:,0]


total_power_5d = np.sum(power_5d,axis=0)
total_power_7d = np.sum(power_7d,axis=0)
freq = np.linspace(0.002,0.016,8)
# print(freq)

plt.figure(dpi=300,figsize=(5,4))
plt.plot(freq,total_power_5d/total_baseline_power_5d-1,'.-',label = '5D')
# plt.plot(freq,total_power_7d/total_baseline_power_7d-1,'.-',label = '7D')
plt.xlabel('frequency [Hz]')
plt.ylabel('normalised power gain')
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 
plt.legend()
plt.grid()
plt.savefig('test.png',bbox_inches='tight')


# plt.plot(baseline_power_7d,',')
# 
# plt.plot(power_7d[:,3])

# plt.savefig('test.png',bbox_inches='tight')

barWidth = 0.25
br1 = np.arange(8)
br2 = [x + barWidth for x in br1]
br3 = [x - barWidth for x in br1]

plt.figure(figsize=(12,8),dpi=300)
plt.rcParams.update({'font.size': 24})
ticks = ['WT'+ str(x+1) for x in range(8)]
plt.bar(br3, baseline_power_7d/baseline_power_7d[0], width = barWidth,edgecolor ='grey', label ='Zero-yaw baseline')
plt.bar(br1, power_7d[:,0]/baseline_power_7d[0], width = barWidth,edgecolor ='grey', label ='$f_{yaw}$=0.002 hz' )
plt.bar(br2, power_7d[:,3]/baseline_power_7d[0], width = barWidth,edgecolor ='grey', label ='$f_{yaw}$=0.008 hz')
plt.xticks(br1,ticks)
plt.ylabel('Normalised power')
plt.legend(fontsize=20)
plt.savefig('power-10-30.png')
