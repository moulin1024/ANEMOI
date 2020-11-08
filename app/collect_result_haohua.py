import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileinput
import sys

# casenum_str = str(sys.argv[1])
# casenum = int(sys.argv[1])
tot_power = np.zeros(117)
for idx in range(117):
    turb_loc = pd.read_csv('../job/fullwake-'+str(idx+1).zfill(3)+'/output/ta_power.csv')
    tot_power[idx] = np.sum(turb_loc['power'])

np.save('../post/haohua/sim/LES_power.npy',tot_power)
