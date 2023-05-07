import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

case = 'HR1-m-0'
file_path = './job/'+case+'/src/output/ta_power.dat'

power = np.loadtxt(file_path)
total_power = np.sum(power)
print(total_power/1e6)
HR_coord = pd.read_csv('./job/'+case+'/input/turb_loc.dat').to_numpy()
print(HR_coord.shape)

plt.figure()
for i in range(HR_coord.shape[0]):
    color_value = (power[i]-np.min(power))/(np.max(power)-np.min(power))
    plt.plot(HR_coord[i,0]/1e3,HR_coord[i,1]/1e3,'o',color=((color_value,0,1)))

plt.axis('scaled')
plt.xlim(0,10.24)
plt.ylim(0,5.12)
plt.show()
