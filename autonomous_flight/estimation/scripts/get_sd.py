# Imports
# ==================================================

import os
import pandas as pd 
import numpy  as np 


# I/O
# ==================================================

os.chdir('D:\\udacity\\drones\\FCND-Estimation-CPP')



# GPS X Data
# ==================================================

data       = pd.read_csv("..\\config\\log\\Graph1.txt")
data_gps_x = data.iloc[:, 1]

print(np.std(data_gps_x))



# Accelerometer X Data
# ==================================================

data       = pd.read_csv("..\\config\\log\\Graph2.txt")
data_acc_x = data.iloc[:, 1]

print(np.std(data_acc_x))