import os
import numpy as np
import pandas as pd
import sys

base_case = sys.argv[1]
target_case = sys.argv[2]
inflow = float(sys.argv[3])

base_case_df = pd.read_csv("./job/"+base_case+"/input/turb_loc.dat")

target_case_df = pd.read_csv("./job/"+target_case+"/input/turb_loc.dat")

target_case_df['yaw'] = base_case_df['yaw'] - inflow

target_case_df.to_csv("./job/"+target_case+"/input/turb_loc.dat",index=False)