import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle

# Load previously generated dataset from DataGenMultivariate.py
file = open('datasetMultivariate.pkl', 'rb')
if not file:
    sys.exit("No datasetMultivariate.pkl file was found")
else:
    dataset = pickle.load(file)
if not dataset:
    sys.exit("Data loaded was empty from datasetMultivariate.pkl file")

# Single Variable Weak Regressor/Learners definition
# Linear model
def LinModel(x, a1, a0):
    return a1*x + a0

# Quadratic model    
def QuadModel(x, a2, a1, a0):
    return a2*(x^2) + a1*x + a0

# Logarithmic model    
def LogModel(x, a1, a0):
    return a1*np.log(x) + a0
    
# Exponential model    
def ExpModel(x, a2, a1, a0):
    return a2*(a1^(x)) + a0
    
# 3rd Order Polynomial model    
def PolyModel(x, a3, a2, a1, a0):
    return a3*(x^3) + a2*(x^2) + a1*x + a0

# ----------------- Weak Regressor System Identification ----------------------------

# Defining constant and variable inputs for Dataset subsampling for 3D visualization
# Performance vs WH and C (filter size k and number of filters N are constant)
k_const = 3
N_const = 228
WH_var = []
C_var = []
LAT_WHC = []
POW_WHC = []
E_WHC = []
T_WHC = []
# Performance vs k and N (Input tensor size WH_in and C_in are constant)
WH_const = 105
C_const = 228
k_var = []
N_var = []
LAT_kN = []
POW_kN = []
E_kN = []
T_kN = []

# Retrieving sample data from Dataset (sample = [WH, C, k, N, LAT, POW, E, T])
for sample in dataset:
    if sample[2] == k_const and sample[3] == N_const:
        WH_var.append(sample[0])
        C_var.append(sample[1])
        LAT_WHC.append(sample[4])
        POW_WHC.append(sample[5])
        E_WHC.append(sample[6])
        T_WHC.append(sample[7])
    elif sample[0] == WH_const and sample[1] == C_const:
        k_var.append(sample[2])
        N_var.append(sample[3])
        LAT_WHC.append(sample[4])
        POW_WHC.append(sample[5])
        E_WHC.append(sample[6])
        T_WHC.append(sample[7])       

# Plot subsampled dataset
# Power vs Input tensor size (P vs WH and C) with kernel size and depth constant (k and N)
plt.figure()
plt.gca(projection='3d')
plt.plot_surface(WH_var, C_var, POW_WHC, rstride=8, cstride=8, alpha=0.3)
plt.contour(WH_var, C_var, POW_WHC, zdir='z', offset=0, cmap=cm.coolwarm)
plt.contour(WH_var, C_var, POW_WHC, zdir='x', offset=0, cmap=cm.coolwarm)
plt.contour(WH_var, C_var, POW_WHC, zdir='y', offset=0, cmap=cm.coolwarm)
plt.title('Power vs Input tensor size', fontsize = 20)
plt.xlabel('Width and Height (HW)', fontsize = 18)
plt.set_xlim(0, 100)
plt.ylabel('Number of Channels (C)', fontsize = 18)
plt.set_ylim(0, 512)
plt.zlabel('Power (W)', fontsize = 18)
plt.set_zlim(0, 8)

# Power vs kernel size and depth variable (P vs k and N) with Input tensor size constant (WH and C) 
plt.figure()
plt.gca(projection='3d')
plt.plot_surface(k_var, N_var , POW_kN, rstride=8, cstride=8, alpha=0.3)
plt.contour(k_var, N_var , POW_kN, zdir='z', offset=0, cmap=cm.coolwarm)
plt.contour(k_var, N_var , POW_kN, zdir='x', offset=0, cmap=cm.coolwarm)
plt.contour(k_var, N_var , POW_kN, zdir='y', offset=0, cmap=cm.coolwarm)
plt.title('Power vs Input tensor size', fontsize = 20)
plt.xlabel('Kernel size (k)', fontsize = 18)
plt.set_xlim(0, 11)
plt.ylabel('Number of filters (N)', fontsize = 18)
plt.set_ylim(0, 512)
plt.zlabel('Power (W)', fontsize = 18)
plt.set_zlim(0, 8)

#plt.legend()
#plt.grid()

plt.show()
