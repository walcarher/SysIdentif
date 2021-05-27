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

# Plot results
plt.figure()
x = np.linspace(1, len(power), len(power))
y = np.asarray(power)
mean = np.mean(y)
std = np.std(y)
y_norm = (y - np.mean(y)) / (np.std(y))
plt.plot(x,y_norm)
y = 1000*np.asarray(powerEst)
y_norm = (y - np.mean(y)) / (np.std(y))
plt.plot(x,y_norm)
plt.grid()
plt.show()
