import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Load parameters to test
# file = open('debugparametersEGPU.pkl', 'rb')
# if not file:
    # sys.exit("No debugparametersEGPU.pkl file was found")
# else:
    # parameters = pickle.load(file)
    # print(parameters)

# Energy KPI estimation function from previous SI parameters
# def EnergyEst(HW, C, k, N):
   # return parameters[0]*(HW**2)+parameters[1]*(HW)+parameters[2]+parameters[3]*(C**3)+parameters[4]*(C**2)+parameters[5]*(C) + \
   # parameters[6]+parameters[7]*(k**3)+parameters[8]*(k**2)+parameters[9]*(k)+parameters[10]+parameters[11]*(N**2)+parameters[12]*(N)+parameters[13]

# Load GPU Power KPI time series to test
file = open('PowerPlot_GPU.pkl', 'rb')
if not file:
    sys.exit("No PowerPlot_GPU.pkl file was found")
else:
    power = pickle.load(file)
    powerEst = pickle.load(file)

# Plot Power results
plt.figure()
x = np.linspace(1, len(power), len(power))
y = np.asarray(power)
mean = np.mean(y)
std = np.std(y)
#y_norm = (y - np.mean(y)) / (np.std(y))
plt.plot(x,y,'k-',linewidth=1, label='Measured Power')
y = 1000*np.asarray(powerEst)
y_norm = (y - np.mean(y)) / (np.std(y))
plt.plot(x,std*y_norm+mean,'g',linestyle='dashed', linewidth=3, label='Estimated Power')
plt.xlabel('Number of Channels (N)')
plt.ylabel('Power (mW)')
#plt.grid()
plt.legend()

# Load GPU Energy KPI time series to test
file = open('EnergyPlot_GPU.pkl', 'rb')
if not file:
    sys.exit("No PowerPlot_GPU.pkl file was found")
else:
    energy = pickle.load(file)
    energyEst = pickle.load(file)

# Remove initialization terms
energy.pop(0)
energyEst.pop(0)
    
# Plot Energy results
plt.figure()
x = np.linspace(1, len(energy), len(energy))
y = np.asarray(energy)
#y = (y - np.min(y)) / (np.max(y)-np.min(y))
mean0 = np.mean(y[0:5000])
std0 = np.std(y[0:5000])
mean1 = np.mean(y[5000:10000])
std1 = np.std(y[5000:10000])
mean2 = np.mean(y[10000:15000])
std2 = np.std(y[10000:15000])
mean3 = np.mean(y[15000:20000])
std3 = np.std(y[15000:20000])
mean4 = np.mean(y[20000:25000])
std4 = np.std(y[20000:25000])

#y_norm = (y - np.mean(y)) / (np.std(y))
plt.plot(x,y,'k-',linewidth=1, label='Measured Energy')
y = np.asarray(energyEst)
y[0:5000] = std0*y[0:5000]+mean0
y[5000:10000] = std1*y[5000:10000]+mean1
y[10000:15000] = std2*y[10000:15000]+mean2
y[15000:20000] = std3*y[15000:20000]+mean3
y[20000:25000] = std4*y[20000:25000]+mean4

#y = (y - np.min(y)) / (np.max(y)-np.min(y))
plt.plot(x,y,'g',linestyle='dashed', linewidth=3, label='Estimated Energy')
#plt.plot(x,y,'g',linestyle='dashed', linewidth=3, label='Estimated Energy')
plt.xlabel('Number of Channels (N)')
plt.ylabel('Energy (mJ)')
#plt.grid()
plt.legend()

plt.show()
