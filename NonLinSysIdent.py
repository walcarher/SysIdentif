import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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
k_const = 11
N_const = 512
WH_var = []
C_var = []
LAT_WHC = []
POW_WHC = []
E_WHC = []
T_WHC = []
# Performance vs k and N (Input tensor size WH_in and C_in are constant)
WH_const = 100
C_const = 512
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
        LAT_kN.append(sample[4])
        POW_kN.append(sample[5])
        E_kN.append(sample[6])
        T_kN.append(sample[7])       

# Plot subsampled dataset for 3D Visualization
# Latency vs Input tensor size (LAT vs WH and C) with kernel size and depth constant (k and N)
fig1 = plt.figure()
X = np.array(WH_var)
Y = np.array(C_var)
Z = np.array(LAT_WHC)
ax1 = fig1.gca(projection='3d')
ax1.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
ax1.tricontour(X, Y, Z, zdir='x', offset=100, cmap=cm.coolwarm)
ax1.tricontour(X, Y, Z, zdir='y', offset=512, cmap=cm.coolwarm)
plt.title('Latency vs Input tensor size')
ax1.set_xlabel('Width and Height (HW)')
#ax.set_xlim(,)
ax1.set_ylabel('Number of Channels (C)')
#ax.set_ylim(,)
ax1.set_zlabel('Latency (ms)')
#ax.set_zlim(,)
# Latency vs Filter size and depth (LAT vs k and N) with input tensor size constant (WH and C)
fig2 = plt.figure()
X = np.array(k_var)
Y = np.array(N_var)
Z = np.array(LAT_kN)
ax2 = fig2.gca(projection='3d')
ax2.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
ax2.tricontour(X, Y, Z, zdir='x', offset=11, cmap=cm.coolwarm)
ax2.tricontour(X, Y, Z, zdir='y', offset=512, cmap=cm.coolwarm)
plt.title('Latency vs Kernel Tensor')
ax2.set_xlabel('Kernel Size (k)')
#ax.set_xlim(,)
ax2.set_ylabel('Number of Filters (N)')
#ax.set_ylim(,)
ax2.set_zlabel('Latency (ms)')
#ax.set_zlim(,)

# Energy vs Input tensor size (E vs WH and C) with kernel size and depth constant (k and N)
fig3 = plt.figure()
X = np.array(WH_var)
Y = np.array(C_var)
Z = np.array(E_WHC)
ax3 = fig3.gca(projection='3d')
ax3.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
ax3.tricontour(X, Y, Z, zdir='x', offset=100, cmap=cm.coolwarm)
ax3.tricontour(X, Y, Z, zdir='y', offset=512, cmap=cm.coolwarm)
plt.title('Energy vs Input tensor size')
ax3.set_xlabel('Width and Height (HW)')
#ax.set_xlim(,)
ax3.set_ylabel('Number of Channels (C)')
#ax.set_ylim(,)
ax3.set_zlabel('Energy (J)')
#ax.set_zlim(,)
# Energy vs Filter size and depth (E vs k and N) with input tensor size constant (WH and C)
fig4 = plt.figure()
X = np.array(k_var)
Y = np.array(N_var)
Z = np.array(E_kN)
ax4 = fig4.gca(projection='3d')
ax4.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
ax4.tricontour(X, Y, Z, zdir='x', offset=11, cmap=cm.coolwarm)
ax4.tricontour(X, Y, Z, zdir='y', offset=512, cmap=cm.coolwarm)
plt.title('Energy vs Kernel Tensor')
ax4.set_xlabel('Kernel Size (k)')
#ax.set_xlim(,)
ax4.set_ylabel('Number of Filters (N)')
#ax.set_ylim(,)
ax4.set_zlabel('Energy (J)')
#ax.set_zlim(,)

# Power vs Input tensor size (P vs WH and C) with kernel size and depth constant (k and N)
fig5 = plt.figure()
X = np.array(WH_var)
Y = np.array(C_var)
Z = np.array(POW_WHC)
ax5 = fig5.gca(projection='3d')
ax5.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
ax5.tricontour(X, Y, Z, zdir='x', offset=100, cmap=cm.coolwarm)
ax5.tricontour(X, Y, Z, zdir='y', offset=512, cmap=cm.coolwarm)
plt.title('Power vs Input tensor size')
ax5.set_xlabel('Width and Height (HW)')
#ax.set_xlim(,)
ax5.set_ylabel('Number of Channels (C)')
#ax.set_ylim(,)
ax5.set_zlabel('Power (W)')
#ax.set_zlim(,)
# Power vs Filter size and depth (P vs k and N) with input tensor size constant (WH and C)
fig6 = plt.figure()
X = np.array(k_var)
Y = np.array(N_var)
Z = np.array(POW_kN)
ax6 = fig6.gca(projection='3d')
ax6.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
ax6.tricontour(X, Y, Z, zdir='x', offset=11, cmap=cm.coolwarm)
ax6.tricontour(X, Y, Z, zdir='y', offset=512, cmap=cm.coolwarm)
plt.title('Power vs Kernel Tensor')
ax6.set_xlabel('Kernel Size (k)')
#ax.set_xlim(,)
ax6.set_ylabel('Number of Filters (N)')
#ax.set_ylim(,)
ax6.set_zlabel('Power (W)')
#ax.set_zlim(,)

# Throughput vs Input tensor size (T vs WH and C) with kernel size and depth constant (k and N)
fig6 = plt.figure()
X = np.array(WH_var)
Y = np.array(C_var)
Z = np.array(T_WHC)
ax6 = fig6.gca(projection='3d')
ax6.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
ax6.tricontour(X, Y, Z, zdir='x', offset=100, cmap=cm.coolwarm)
ax6.tricontour(X, Y, Z, zdir='y', offset=512, cmap=cm.coolwarm)
plt.title('Throughput vs Input tensor size')
ax6.set_xlabel('Width and Height (HW)')
#ax.set_xlim(,)
ax6.set_ylabel('Number of Channels (C)')
#ax.set_ylim(,)
ax6.set_zlabel('Throughput (GB/s)')
#ax.set_zlim(,)
# Throughput vs Filter size and depth (T vs k and N) with input tensor size constant (WH and C)
fig7 = plt.figure()
X = np.array(k_var)
Y = np.array(N_var)
Z = np.array(T_kN)
ax7 = fig7.gca(projection='3d')
ax7.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
ax7.tricontour(X, Y, Z, zdir='x', offset=11, cmap=cm.coolwarm)
ax7.tricontour(X, Y, Z, zdir='y', offset=512, cmap=cm.coolwarm)
plt.title('Throughput vs Kernel Tensor')
ax7.set_xlabel('Kernel Size (k)')
#ax.set_xlim(,)
ax7.set_ylabel('Number of Filters (N)')
#ax.set_ylim(,)
ax7.set_zlabel('Throughput (GB/s)')
#ax.set_zlim(,)

#plt.legend()
#plt.grid()

plt.show()
