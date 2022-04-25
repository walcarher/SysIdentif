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

##############################################################CPU#######################################################

# Load CPU parameters to test
file = open('parametersECPU.pkl', 'rb')
if not file:
    sys.exit("No parametersECPU.pkl file was found")
else:
    parametersCPU = pickle.load(file)
    
# Load GPU parameters to test
file = open('parametersEGPU.pkl', 'rb')
if not file:
    sys.exit("No parametersEGPU.pkl file was found")
else:
    parametersGPU = pickle.load(file)
    
# Load parameters to test
file = open('parametersEFPGA.pkl', 'rb')
if not file:
    sys.exit("No parametersEFPGA.pkl file was found")
else:
    parametersFPGA = pickle.load(file)

# Models names evaluated
models = ['VGG16', 'InceptionV3', 'SqueezeNet']   

# Generate parameters for VGG16 CNN Layer 1  
WH = np.array([224,224,299])
C =  np.array([3  ,  3,  3])
k =  np.array([3  ,  3,  3])
N =  np.array([64 , 32, 64])

x = np.arange(len(models))

energyEstCPU = 1000*EnergyEstCPU([WH,C,k,N],*parametersCPU) # in mJ
energyEstGPU = 1000*EnergyEstGPU([WH,C,k,N],*parametersGPU) # in mJ
energyEstFPGA = EnergyEstFPGA([WH,C,k,N],*parametersFPGA)/1000000 # in mJ
plt.bar(x -0.22, energyEstCPU, width=0.2, fill=False, edgecolor='red', linestyle='dashed', linewidth=2, label='Estimated CPU Energy')
plt.bar(x, energyEstGPU, width=0.2, fill=False, edgecolor='darkgreen', linestyle='dashed', linewidth=2, label='Estimated GPU Energy')
plt.bar(x +0.32, energyEstFPGA, width=0.4, fill=False, edgecolor='navy', linestyle='dashed', linewidth=2, label='Estimated FPGA Energy')
plt.xticks(x, models)
plt.ylabel('Energy (mJ)')

#plt.grid()
plt.legend()

plt.show()
