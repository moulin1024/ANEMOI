import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileinput
import sys

# casenum_str = str(sys.argv[1])
# casenum = int(sys.argv[1])
tot_power = np.zeros(56)
for idx in range(56):
    turb_loc = pd.read_csv('../job/fullwake-lambda-'+str(idx+1).zfill(3)+'/output/ta_power.csv')
    tot_power[idx] = np.sum(turb_loc['power'])

np.save('../post/haohua/sim/LES_power-lambda.npy',tot_power)
