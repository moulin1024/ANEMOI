import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

jobname = 'ultralong-0'
data = pd.read_csv('job/'+jobname+'/src/output/root_moment.csv',header=None).to_numpy()
print(data.shape)
plt.plot(data)
plt.savefig('test.png')