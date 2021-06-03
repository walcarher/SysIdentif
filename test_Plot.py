import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

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
y = (y - np.min(y)) / (np.max(y)-np.min(y))
mean = np.mean(y)
std = np.std(y)
#y_norm = (y - np.mean(y)) / (np.std(y))
plt.plot(x,y,'k-',linewidth=1, label='Measured Energy')
y = energyEst
y = (y - np.min(y)) / (np.max(y)-np.min(y))
plt.plot(x,y,'g',linestyle='dashed', linewidth=3, label='Estimated Energy')
#plt.plot(x,y,'g',linestyle='dashed', linewidth=3, label='Estimated Energy')
plt.xlabel('Number of Channels (N)')
plt.ylabel('Energy (mJ)')
#plt.grid()
plt.legend()

plt.show()
