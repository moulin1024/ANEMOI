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

static_yaw_power = np.zeros([3,8])
baseline_power = pd.read_csv('../job/dyn-yaw-8wt-baseline/src/output/ta_power.dat',header=None).to_numpy()[:,0]
for i in range(3):
    static_yaw_power[i,:] = pd.read_csv('../job/dyn-yaw-8wt-'+str((i+1)*10)+'-static/src/output/ta_power.dat',header=None).to_numpy()[:,0]


totalpower = np.sum(power,axis=-1)/np.sum(baseline_power)-1
total_static_yaw_power = np.sum(static_yaw_power,axis=-1)/np.sum(baseline_power)-1
period = [300,400,500,600]


plt.figure(figsize=(8,6),dpi=300)
plt.rcParams.update({'font.size': 14})
# plt.scatter(300,totalpower.T[0,2],marker='x',s=200,c='r')
for j in range(3):
    plt.plot(period,totalpower.T[:,j],'o-',label='$\gamma_{max}=$'+str((j+1)*10)+'$ ^\circ$')

# plt.gca().set_prop_cycle(None)
# for j in range(3):
#     plt.plot([300,600],[total_static_yaw_power[j],total_static_yaw_power[j]],'--',label='Static yaw: $\gamma=$'+str((j+1)*10)+'$ ^\circ$')
# # print(totalpower)

plt.legend()
plt.ylim([-0.01,0.05])
plt.ylabel('Normalied total power gain')
plt.xlabel('Cycle period (s)')
plt.savefig('total_power_dyn.png')




barWidth = 0.25
br1 = np.arange(8)
br2 = [x + barWidth/2 for x in br1]
br3 = [x - barWidth/2 for x in br1]

plt.figure(figsize=(12,8),dpi=300)
plt.rcParams.update({'font.size': 24})
ticks = ['WT'+ str(x+1) for x in range(8)]
plt.bar(br3, baseline_power/baseline_power[0], width = barWidth,edgecolor ='grey', label ='Zero-yaw baseline')
# plt.bar(br1, static_yaw_power[2,:]/1e6, width = barWidth,edgecolor ='grey', label ='Static yaw ($\gamma = 30^\circ$)')
plt.bar(br2, power[2,1,:]/baseline_power[0], width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 30^\circ$, period 400 s')
plt.xticks(br1,ticks)
plt.ylabel('Normalised power')
plt.legend(fontsize=20)
plt.savefig('power-30.png')

print(power[2,1,:]/baseline_power[0])
print(baseline_power/baseline_power[0])
baseline_power/baseline_power[0]