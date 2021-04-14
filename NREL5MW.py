import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# import seaborn as sns
from datetime import datetime, date, time
import warnings
from scipy import interpolate
import os
warnings.filterwarnings('ignore')

def get_CLCD(airfoil_name):
    DATA_FOLDER = 'DATA/'

    DATASET1 = DATA_FOLDER + "Drag DU21-A17_pi.csv"
    DATASET2 = DATA_FOLDER + "Lift DU21-A17_pi.csv"

    #Load of the compressed files in a specific way in function of the type of data file 
    df_drag = pd.read_csv(DATASET1,sep =';',decimal=",")
    df_lift = pd.read_csv(DATASET2,sep =';',decimal=",")

    #Defining the columns names of the uploaded dataframe
    df_drag.columns=["Angle_of_attack", "Cd"]
    df_lift.columns=["Angle_of_attack", "Cl"]

    x1 = df_drag["Angle_of_attack"].astype(float, errors = 'raise')
    y1 = df_drag["Cd"].astype(float, errors = 'raise')
    f1 = interpolate.interp1d(x1, y1,fill_value="extrapolate")

    x2 = df_lift["Angle_of_attack"].astype(float, errors = 'raise')
    y2 = df_lift["Cl"].astype(float, errors = 'raise')
    f2 = interpolate.interp1d(x2, y2,fill_value="extrapolate")

    x_interp = np.linspace(-180,180,361)
    # print(x_interp)
    CL = f1(x_interp)
    CD = f2(x_interp)
    return CL,CD


airfoil_list = ['DU40-A17','DU35-A17','DU30-A17','DU25-A17','DU21-A17', 'NACA64-A17']
CL_mat = np.zeros([361,6])
CD_mat = np.zeros([361,6])
for idx,name in enumerate(airfoil_list):
    CL, CD = get_CLCD(name)
    CL_mat[:,idx] = np.array(CL)
    CD_mat[:,idx] = np.array(CD)
    # CD_mat[:,idx] = CD
    # print(CL.shape)
    # print(CD.size)
    
np.savetxt('CL_NREL.csv', CL_mat, delimiter=',')
np.savetxt('CD_NREL.csv', CD_mat, delimiter=',')

# CL,CD = get_CLCD(airfoil_name)
# print(CL.shape)
