import os
import numpy as np
import pandas as pd
import sys
theta = sys.argv[1]

theta = np.radians(np.double(theta))

wind_df = pd.read_csv("HornsRev.dat")
x = wind_df["x"]
y = wind_df["y"]

x_rotate = x[0] + (x-x[0])*np.cos(theta) - (y-y[0]) * np.sin(theta)
y_rotate = y[0] + (y-y[0])*np.cos(theta) + (x-x[0]) * np.sin(theta)

wind_df["x"] = x_rotate
wind_df["y"] = y_rotate
wind_df.to_csv("turb_loc.dat",index=False)

