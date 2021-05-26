import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Load parameters to test
file = open('parametersGPU.pkl', 'rb')
if not file:
    sys.exit("No parametersGPU.pkl file was found")
else:
    parameters = pickle.load(file)

HW = 200
C = 1
k_s = np.concatenate((np.full((1,100), 1), np.full((1,100), 3), np.full((1,100), 5), np.full((1,100), 7), np.full((1,100), 11)), axis=1)
k = np.concatenate((k_s,k_s,k_s,k_s,k_s),axis=1)
N = np.concatenate((np.full_like(k_s, 100),np.full_like(k_s, 200),np.full_like(k_s, 300),np.full_like(k_s, 400),np.full_like(k_s, 500)),axis=1)


plt.figure()
x = np.linspace(1, N.size, N.size)
y = parameters[0]*(HW**3)+parameters[1]*(HW**2)+parameters[2]*(HW)+parameters[3]+parameters[4]*np.log(C)+parameters[5] + \
    parameters[6]*(k**2)+parameters[7]*(k)+parameters[8]+parameters[9]*np.log(N)+parameters[10]+parameters[11]
y = y.ravel()
plt.grid()

plt.plot(x,y)
plt.show()
