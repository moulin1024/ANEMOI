import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('job/NREL-m/src/output/root_moment.csv',header=None).to_numpy()
plt.plot(data)
plt.xlim([900,2000])
plt.show()