import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py


power = np.zeros([3,4,8])
for i in range(3):
    for j in range(4):
        casename = 'dyn-yaw-8wt-'+str((i+1)*10)+'-'+str((j+3)*100)+'s'
        filepath = '../job/'+casename+'/src/output/ta_power.dat'
        # print(filepath)
        # print(casename)
        power[i,j,:] = pd.read_csv(filepath,header=None).to_numpy()[:,0]
        # print(power)
        # filepath = 

period_10 = [80,100,120,140,150,200,300,400,500,600]
yaw_power_10 = np.zeros([len(period_10),8])
yaw_power_20 = np.zeros([5,8])
yaw_power_30 = np.zeros([4,8])


baseline_power = pd.read_csv('../job/dyn-yaw-8wt-baseline/src/output/ta_power.dat',header=None).to_numpy()[:,0]


for i in range(len(period_10)):
    yaw_power_10[i,:] = pd.read_csv('../job/dyn-yaw-8wt-10-'+str(int(period_10[i]))+'s/src/output/ta_power.dat',header=None).to_numpy()[:,0]

for i in range(5):
    yaw_power_20[i,:] = pd.read_csv('../job/dyn-yaw-8wt-20-'+str((i+2)*100)+'s/src/output/ta_power.dat',header=None).to_numpy()[:,0]

for i in range(4):
    yaw_power_30[i,:] = pd.read_csv('../job/dyn-yaw-8wt-30-'+str((i+3)*100)+'s/src/output/ta_power.dat',header=None).to_numpy()[:,0]


totalpower = np.sum(power,axis=-1)/np.sum(baseline_power)-1
total_yaw_power_10 = np.sum(yaw_power_10,axis=-1)/np.sum(baseline_power)-1
total_yaw_power_20 = np.sum(yaw_power_20,axis=-1)/np.sum(baseline_power)-1
total_yaw_power_30 = np.sum(yaw_power_30,axis=-1)/np.sum(baseline_power)-1

# period_10 = [100,140,200,300,400,500,600]
period = [100,200,300,400,500,600]

print(yaw_power_10)
plt.figure(figsize=(8,6),dpi=300)
plt.rcParams.update({'font.size': 14})
# plt.scatter(300,totalpower.T[0,2],marker='x',s=200,c='r')
# for j in range(3):
plt.plot(period_10,total_yaw_power_10,'o-',label='$\gamma_{max}=10^\circ$')
plt.plot(period[1:],total_yaw_power_20,'o-',label='$\gamma_{max}=20^\circ$')
plt.plot(period[2:],total_yaw_power_30,'o-',label='$\gamma_{max}=30^\circ$')

# plt.gca().set_prop_cycle(None)
# for j in range(3):
#     plt.plot([300,600],[total_static_yaw_power[j],total_static_yaw_power[j]],'--',label='Static yaw: $\gamma=$'+str((j+1)*10)+'$ ^\circ$')
# # print(totalpower)

plt.legend()
plt.ylim([-0.01,0.06])
plt.ylabel('Normalied total power gain')
plt.xlabel('Cycle period (s)')
plt.savefig('total_power_dyn_10.png')




barWidth = 0.25
br1 = np.arange(8)
br2 = [x + barWidth for x in br1]
br3 = [x - barWidth for x in br1]

plt.figure(figsize=(12,8),dpi=300)
plt.rcParams.update({'font.size': 24})
ticks = ['WT'+ str(x+1) for x in range(8)]
plt.bar(br3, baseline_power/baseline_power[0], width = barWidth,edgecolor ='grey', label ='Zero-yaw baseline')
plt.bar(br1, yaw_power_10[0,:]/baseline_power[0], width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 10^\circ$, period 100 s')
plt.bar(br2, yaw_power_30[1,:]/baseline_power[0], width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 30^\circ$, period 400 s')
plt.xticks(br1,ticks)
plt.ylabel('Normalised power')
plt.legend(fontsize=20)
plt.savefig('power-10-30.png')

print(power[2,1,:]/baseline_power[0])
print(baseline_power/baseline_power[0])
baseline_power/baseline_power[0]