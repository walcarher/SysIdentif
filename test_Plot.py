import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Linear model
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

# CPU Energy KPI estimation function from previous SI parameters    
def EnergyEstCPU(x ,*b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = LinModel(x[1],b[3],b[4])
    k_model = PolyModel(x[2],b[5],b[6],b[7],b[8])
    N_model = QuadModel(x[3],b[9],b[10],b[11])
    return HW_model * C_model * k_model * N_model
    
# GPU Energy KPI estimation function from previous SI parameters
def EnergyEstGPU(x ,*b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = PolyModel(x[1],b[3],b[4],b[5],b[6])
    k_model = PolyModel(x[2],b[7],b[8],b[9],b[10])
    N_model = QuadModel(x[3],b[11],b[12],b[13])
    return HW_model * C_model * k_model * N_model
    
# FPGA Energy KPI estimation function from previous SI parameters
def EnergyEstFPGA(x ,*b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = QuadModel(x[1],b[3],b[4],b[5])
    k_model = LinModel(x[2],b[6],b[7])
    N_model = QuadModel(x[3],b[8],b[9],b[10])
    return HW_model * C_model * k_model * N_model

# Load GPU Power KPI time series to test
# file = open('PowerPlot_GPU.pkl', 'rb')
# if not file:
    # sys.exit("No PowerPlot_GPU.pkl file was found")
# else:
    # power = pickle.load(file)
    # powerEst = pickle.load(file)

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

##############################################################CPU#######################################################
# Load CPU Energy KPI time series to test
file = open('EnergyPlot_CPU.pkl', 'rb')
if not file:
    sys.exit("No EnergyPlot_CPU.pkl file was found")
else:
    energyCPU    = pickle.load(file)

# Repeat measurements to match GPU's iterations
energyCPU = np.repeat(np.asarray(energyCPU),10).tolist()
len(energyCPU)
# Remove initialization terms
energyCPU.pop(0)
    
# Plot Energy results
plt.figure()
x = np.linspace(1, len(energyCPU), len(energyCPU))
y = np.asarray(energyCPU)
plt.plot(x,y,'-',color='coral',linewidth=1, label='Measured Energy')

# Load parameters to test
file = open('parametersECPU.pkl', 'rb')
if not file:
    sys.exit("No parametersECPU.pkl file was found")
else:
    parametersCPU = pickle.load(file)

# Generate synthetic data for features    
WH = 32*np.ones(25000)
C = 100*np.ones(25000)
k = np.tile(np.concatenate([np.ones(1000),3*np.ones(1000),5*np.ones(1000),7*np.ones(1000),11*np.ones(1000)]),5)
N = np.concatenate([200*np.ones(5000),300*np.ones(5000),400*np.ones(5000),500*np.ones(5000),600*np.ones(5000)])

energyEstCPU = EnergyEstCPU([WH,C,k,N],*parametersCPU)
energyEstCPU = np.delete(energyEstCPU,0)
y = np.asarray(energyEstCPU)
plt.plot(x,1000*y,color='red',linestyle='dashed', linewidth=3, label='Estimated CPU Energy')
plt.xlabel('Number of Channels (N)')
plt.ylabel('Energy (mJ)')
##############################################################GPU#######################################################
# Load GPU Energy KPI time series to test
file = open('EnergyPlot_GPU.pkl', 'rb')
if not file:
    sys.exit("No EnergyPlot_GPU.pkl file was found")
else:
    energyGPU    = pickle.load(file)

# Remove initialization terms
energyGPU.pop(0)
    
# Plot Energy results
x = np.linspace(1, len(energyGPU), len(energyGPU))
y = np.asarray(energyGPU)
plt.plot(x,y,'-',color='mediumseagreen',linewidth=1, label='Measured Energy')

# Load parameters to test
file = open('parametersEGPU.pkl', 'rb')
if not file:
    sys.exit("No parametersEGPU.pkl file was found")
else:
    parametersGPU = pickle.load(file)

energyEstGPU = EnergyEstGPU([WH,C,k,N],*parametersGPU)
energyEstGPU = np.delete(energyEstGPU,0)
y = np.asarray(energyEstGPU)
plt.plot(x,1000*y, color='darkgreen',linestyle='dashed', linewidth=3, label='Estimated GPU Energy')
plt.xlabel('Number of Channels (N)')
plt.ylabel('Energy (mJ)')

##############################################################FPGA#######################################################

# Load parameters to test
file = open('parametersEFPGA.pkl', 'rb')
if not file:
    sys.exit("No parametersEFPGA.pkl file was found")
else:
    parametersFPGA = pickle.load(file)

energyEstFPGA = EnergyEstFPGA([WH,C,k,N],*parametersFPGA)
energyEstFPGA = np.delete(energyEstFPGA,0)
y = np.asarray(energyEstFPGA)
plt.plot(x,y/1000000, color='navy',linestyle='dashed', linewidth=3, label='Estimated FPGA Energy')
plt.xlabel('Number of Channels (N)')
plt.ylabel('Energy (mJ)')
#plt.grid()
plt.legend()

plt.show()
