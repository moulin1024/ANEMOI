import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileinput
import sys

casenum_str = str(sys.argv[1])
casenum = int(sys.argv[1])

df_full = pd.read_csv('../post/haohua/exp/fullwake_rpm_power_exp_positive.csv',header=None)

turb_loc = pd.read_csv('../job/fullwake-lambda-'+casenum_str+'/input/turb_loc.dat')

turb_loc['gamma'] = df_full.iloc[casenum-1,0:3]
turb_loc.to_csv('../job/fullwake-lambda-'+casenum_str+'/input/turb_loc.dat',index=False)
rpm = df_full.iloc[casenum-1,3:6].to_numpy()

rpm_string = np.array2string(rpm, precision=3, separator=',',formatter={'float_kind':lambda x: "%.1f" % x})
rpm_string = np.char.replace(rpm_string,'[','(/')
rpm_string = np.char.replace(rpm_string,']','/)')
fin = open("../job/fullwake-lambda-"+casenum_str+"/input/config", "rt")
#read file contents to string
data = fin.read()
#replace all occurrences of the required string
data = data.replace('$turb_w', rpm_string)
#close the input file
fin.close()
#open the input file in write mode
fin = open("../job/fullwake-lambda-"+casenum_str+"/input/config", "wt")
#overrite the input file with the resulting data
fin.write(data)
#close the file
fin.close()