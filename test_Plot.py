import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

def LinModel(x, a1, a0):
    return a1*x + a0

# Quadratic model
def QuadModel(x, a2, a1, a0):
    return a2*np.power(x, 2) + a1*x + a0

# Logarithmic model
def LogModel(x, a1, a0):
    return a1*np.log(x) + a0
    
# Exponential model   
def ExpModel(x, a2, a1, a0):
    return a2*np.power(a1, x) + a0
    
# Polynomial model   
def PolyModel(x, a3, a2, a1, a0):
    return a3*np.power(x, 3) + a2*np.power(x, 2) + a1*x + a0

# Energy KPI estimation function from previous SI parameters
def EnergyEst(x ,*b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = LinModel(x[1],b[3],b[4])
    k_model = QuadModel(x[2],b[5],b[6],b[7])
    N_model = LinModel(x[3],b[8],b[9])
    return (HW_model * C_model) * (k_model * N_model)

# Load GPU Power KPI time series to test
file = open('PowerPlot_GPU.pkl', 'rb')
if not file:
    sys.exit("No PowerPlot_GPU.pkl file was found")
else:
    power = pickle.load(file)
    powerEst = pickle.load(file)

# Plot Power results
# plt.figure()
# x = np.linspace(1, len(power), len(power))
# y = np.asarray(power)
# mean = np.mean(y)
# std = np.std(y)
# #y_norm = (y - np.mean(y)) / (np.std(y))
# plt.plot(x,y,'k-',linewidth=1, label='Measured Power')
# y = 1000*np.asarray(powerEst)
# y_norm = (y - np.mean(y)) / (np.std(y))
# plt.plot(x,y_norm+mean,'g',linestyle='dashed', linewidth=3, label='Estimated Power')
# plt.xlabel('Number of Channels (N)')
# plt.ylabel('Power (mW)')
# #plt.grid()
# plt.legend()

# Load GPU Energy KPI time series to test
file = open('EnergyPlot_GPU.pkl', 'rb')
if not file:
    sys.exit("No EnergyPlot_GPU.pkl file was found")
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
min = np.min(y)
max = np.max(y)
#y = (y - min) / (max-min)

#y_norm = (y - np.mean(y)) / (np.std(y))
plt.plot(x,y,'k-',linewidth=1, label='Measured Energy')
#y = np.asarray(energyEst)

#y = (y - np.min(y)) / (np.max(y)-np.min(y))
#plt.plot(x,y,'g',linestyle='dashed', linewidth=3, label='Estimated Energy')
#plt.plot(x,y,'g',linestyle='dashed', linewidth=3, label='Estimated Energy')
#plt.xlabel('Number of Channels (N)')
#plt.ylabel('Energy (mJ)')

# Load parameters to test
file = open('debugparametersEGPU.pkl', 'rb')
if not file:
    sys.exit("No debugparametersEGPU.pkl file was found")
else:
    parameters = pickle.load(file)
    print(parameters)

# Generate synthetic data for features    
WH = 32*np.ones(25000)
C = 100*np.ones(25000)
k = np.tile(np.concatenate([np.ones(1000),3*np.ones(1000),5*np.ones(1000),7*np.ones(1000),11*np.ones(1000)]),5)
N = np.concatenate([200*np.ones(5000),300*np.ones(5000),400*np.ones(5000),500*np.ones(5000),600*np.ones(5000)])

energyEst = EnergyEst([WH,C,k,N],*parameters)
energyEst = np.delete(energyEst,0)
y = np.asarray(energyEst)
y = (y - np.min(y)) / (np.max(y)-np.min(y))
plt.plot(x,(max-min)*y+min,'g',linestyle='dashed', linewidth=3, label='Estimated GPU Energy')
#plt.plot(x,y,'g',linestyle='dashed', linewidth=3, label='Estimated Energy')
plt.xlabel('Number of Channels (N)')
plt.ylabel('Energy (mJ)')


#plt.grid()
plt.legend()

plt.show()
