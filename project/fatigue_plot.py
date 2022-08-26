import numpy as np
import matplotlib.pyplot as plt 
import fatpack
import h5py

data = np.load('DEL_results.npz')
print(data.files)

DEL_baseline = data['DEL_baseline']
DEL_dyn = data['DEL_dyn']
DEL_static = data['DEL_static']
print(DEL_static.shape)

period = [300,400,500,600]

total_DEL = np.sum(DEL_dyn[:,:,:,0],axis=-1)
total_DEL_static = np.sum(DEL_static[:,:,0],axis=-1)
total_DEL_ref = np.sum(DEL_baseline[:,0])

plt.figure(figsize=(8,6),dpi=300)
plt.rcParams.update({'font.size': 14})
for j in range(3):
    plt.plot(period,total_DEL[j,:]/total_DEL_ref-1,'.-',label='$\gamma_{max}=$'+str((j+1)*10)+'$ ^\circ$')
# plt.gca().set_prop_cycle(None)
# for j in range(3):
#     plt.plot([300,600],[total_DEL_static[j],total_DEL_static[j]]/total_DEL_ref-1,'--',label='Static yaw: $\gamma=$'+str((j+1)*10)+'$ ^\circ$')
plt.legend()
plt.ylim([-0.02,0.06])
plt.ylabel('Normalied total DEL increase')
plt.xlabel('Cycle period (s)')
plt.savefig('total_DEL.png')

barWidth = 0.25
plt.figure(figsize=(12,8),dpi=300)
plt.rcParams.update({'font.size': 24})
ticks = ['WT'+ str(x+1) for x in range(8)]
br1 = np.arange(8)
br2 = [x + barWidth/2 for x in br1]
br3 = [x - barWidth/2 for x in br1]
plt.bar(br3, DEL_baseline[:,0]/DEL_baseline[0,0], width = barWidth,edgecolor ='grey', label ='Zero-yaw baseline')
# plt.bar(br1, DEL_dyn[2,0,:,0]/DEL_baseline[0,0], width = barWidth,edgecolor ='grey', label ='Cyclic yaw ($\gamma_{max} = 30^\circ$, period 500 s)')
plt.bar(br2, DEL_dyn[2,1,:,0]/DEL_baseline[0,0], width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 30^\circ$, period 400 s')
plt.xticks(br1,ticks)
plt.ylabel('Normalised DEL')
plt.ylim([0,1.6])
plt.legend(fontsize=20)
plt.savefig('DEL-30.png')


total_DEL = np.sum(DEL_dyn[:,:,:,1],axis=-1)
total_DEL_static = np.sum(DEL_static[:,:,1],axis=-1)
total_DEL_ref = np.sum(DEL_baseline[:,1])

plt.figure(figsize=(8,6),dpi=300)
plt.rcParams.update({'font.size': 14})
for j in range(3):
    plt.plot(period,total_DEL[j,:]/total_DEL_ref-1,'.-',label='$\gamma_{max}=$'+str((j+1)*10)+'$ ^\circ$')
# plt.gca().set_prop_cyclea(None)
# for j in range(3):
#     plt.plot([300,600],[total_DEL_static[j],total_DEL_static[j]]/total_DEL_ref-1,'--',label='Static yaw: $\gamma=$'+str((j+1)*10)+'$ ^\circ$')
plt.legend()
plt.ylim([-0.02,0.06])
plt.ylabel('Normalied total DEL increase')
plt.xlabel('Cycle period (s)')
plt.savefig('total_DEL_yaw.png')

barWidth = 0.25
plt.figure(figsize=(12,8),dpi=300)
plt.rcParams.update({'font.size': 24})
ticks = ['WT'+ str(x+1) for x in range(8)]
br1 = np.arange(8)
br2 = [x + barWidth/2 for x in br1]
br3 = [x - barWidth/2 for x in br1]
plt.bar(br3, DEL_baseline[:,1]/DEL_baseline[0,1], width = barWidth,edgecolor ='grey', label ='Zero-yaw baseline')
# plt.bar(br1, DEL_static[0,:,1]/DEL_baseline[0,1], width = barWidth,edgecolor ='grey', label ='Static yaw ($\gamma = 30^\circ$)')
# plt.bar(br1, DEL_dyn[2,0,:,1]/DEL_baseline[0,1], width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 30^\circ$, period 300 s')
plt.bar(br2, DEL_dyn[2,1,:,1]/DEL_baseline[0,1], width = barWidth,edgecolor ='grey', label ='$\gamma_{max} = 30^\circ$, period 400 s')
plt.xticks(br1,ticks)
plt.ylabel('Normalised DEL')
plt.ylim([0,1.6])
plt.legend(fontsize=20)
plt.savefig('DEL-30-yaw.png')
