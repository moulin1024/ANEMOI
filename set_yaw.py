import os
import numpy as np
import pandas as pd
import sys
case = sys.argv[1]
yaw1 = sys.argv[2]
yaw2 = sys.argv[3]
yaw_angle = np.asarray([yaw1,yaw2,0])
wind_df = pd.read_csv("turb_loc.dat")
wind_df["yaw"] = yaw_angle
wind_df.to_csv("job/"+case+"/input/turb_loc.dat",index=False)
