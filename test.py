import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

jobname = 'NREL-m-long'
data = pd.read_csv('job/'+jobname+'/src/output/power.csv',header=None).to_numpy()
print(data.shape[0])
power = np.reshape(data[0:3*20000],[20000,3])
t = np.asarray(range(200))
plt.plot(t,power[::100,1])
plt.plot(t,power[::100,0])
plt.plot(t,power[::100,2])
plt.savefig('test.png')