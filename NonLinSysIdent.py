import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import pickle

# Argument configuration
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_plot", type = int, choices=[0, 1],
		 help = "Shows Dataset subsampling used for the System Identification",
		 default = 0)
parser.add_argument("-r", "--result_plot", type = int, choices=[0, 1],
		 help = "Shows resulting plots from the best selected models",
		 default = 0)
args = parser.parse_args()

# Load previously generated dataset from DataGenMultivariate.py
file = open('datasetMultivariate.pkl', 'rb')
if not file:
    sys.exit("No datasetMultivariate.pkl file was found")
else:
    dataset = pickle.load(file)
if not dataset:
    sys.exit("Data loaded was empty from datasetMultivariate.pkl file")

# Single Variable Weak Regressor/Learners definition

class Name:
    def __init__(self, r):
        self.name = r

    def __call__(self, f):
        f.name = self.name
        return f

# Linear model
@Name('Linear')
def LinModel(x, a1, a0):
    return a1*x + a0

# Quadratic model
@Name('Quadratic')
def QuadModel(x, a2, a1, a0):
    return a2*np.power(x, 2) + a1*x + a0

# Logarithmic model
@Name('Logarithmic')  
def LogModel(x, a1, a0):
    return a1*np.log(x) + a0
    
# Exponential model   
@Name('Exponential') 
def ExpModel(x, a2, a1, a0):
    return a2*np.power(a1, x) + a0
    
# Polynomial model   
@Name('Polynomial') 
def PolyModel(x, a3, a2, a1, a0):
    return a3*np.power(x, 3) + a2*np.power(x, 2) + a1*x + a0

# ----------------- Weak Regressor System Identification ----------------------------

# Defining constant and variable inputs for Dataset subsampling for 3D visualization
# Performance vs WH and C (filter size k and number of filters N are constant)
k_const = 5
N_const = 284
WH_var, C_var, LAT_WHC, POW_WHC, E_WHC, T_WHC = [],[],[],[],[],[]

# Performance vs k and N (Input tensor size WH_in and C_in are constant)
WH_const = 70
C_const = 285
k_var, N_var, LAT_kN, POW_kN, E_kN, T_kN = [],[],[],[],[],[]

# Retrieving sample data from Dataset (sample = [WH, C, k, N, LAT, POW, E, T])
for sample in dataset:
    if sample[2] == k_const and sample[3] == N_const:
        WH_var.append(sample[0])
        C_var.append(sample[1])
        LAT_WHC.append(sample[4])
        POW_WHC.append(sample[5])
        E_WHC.append(sample[6])
        T_WHC.append(sample[7])
    if sample[0] == WH_const and sample[1] == C_const:
        k_var.append(sample[2])
        N_var.append(sample[3])
        LAT_kN.append(sample[4])
        POW_kN.append(sample[5])
        E_kN.append(sample[6])
        T_kN.append(sample[7])       


# Plot subsampled dataset for 3D Visualization if data_plot is enabled
if args.data_plot:
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

# Defining constant and variable inputs for Dataset subsampling for SI
WH_var, LAT_WH, POW_WH, E_WH, T_WH = [],[],[],[],[]
C_var, LAT_C, POW_C, E_C, T_C = [],[],[],[],[]
k_var, LAT_k, POW_k, E_k, T_k = [],[],[],[],[]
N_var, LAT_N, POW_N, E_N, T_N = [],[],[],[],[]

# Retrieving sample data from Dataset for SI(sample = [WH, C, k, N, LAT, POW, E, T])
for sample in dataset:
    if  sample[1] == C_const and sample[2] == k_const and sample[3] == N_const:
        WH_var.append(sample[0])
        LAT_WH.append(sample[4])
        POW_WH.append(sample[5])
        E_WH.append(sample[6])
        T_WH.append(sample[7])
    if sample[0] == WH_const and sample[2] == k_const and sample[3] == N_const:
        C_var.append(sample[1])
        LAT_C.append(sample[4])
        POW_C.append(sample[5])
        E_C.append(sample[6])
        T_C.append(sample[7])
    if sample[0] == WH_const and sample[1] == C_const and sample[3] == N_const:
        k_var.append(sample[2])
        LAT_k.append(sample[4])
        POW_k.append(sample[5])
        E_k.append(sample[6])
        T_k.append(sample[7])
    if sample[0] == WH_const and sample[1] == C_const and sample[2] == k_const:
        N_var.append(sample[3])
        LAT_N.append(sample[4])
        POW_N.append(sample[5])
        E_N.append(sample[6])
        T_N.append(sample[7]) 
        
# Lists of KPIs per feature
kpis_variable = [[LAT_WH, LAT_C, LAT_k, LAT_N], [POW_WH, POW_C, POW_k, POW_N], [E_WH, E_C, E_k, E_N], [T_WH, T_C, T_k, T_N]]
# List of features
features = [WH_var, C_var, k_var, N_var]
# Previously defined models
Models = [LinModel, QuadModel, LogModel, ExpModel, PolyModel]
# Obtained optimal parameters 
parameters = []
# Obtained RMSE
rmses = []

# # Models for each feature by keeping all others features constant using LM method for curve fitting 
for kpis in kpis_variable:
    for feature, kpi in zip(features, kpis):
        for Model in Models:
            parameter, covariance = curve_fit(Model, feature, kpi, maxfev=5000)
            parameters.append(parameter)
            # Computing RMSE
            rmse = np.sqrt(np.sum((kpi - Model(np.asarray(feature), *parameter)) ** 2)/(len(kpi) - len(parameter)))
            rmses.append(rmse)


if args.result_plot:
    # Ordered KPI names
    kpi_names = ['Latency', 'Power', 'Energy', 'Throughput']
    # with units
    kpi_units = ['ms', 'W', 'J', 'GB/s']
    # Ordered feature names
    feature_names = ['Input Tensor Size', 'Input Tensor Depth', 'Kernel size', 'Number of Kernel Filters']
    # with 
    feature_symbol = ['WH', 'C', 'k', 'N']
    # plot colors and markers configurations
    configs = ['r-', 'cs-', 'm^-', 'bD-', 'yp-']
    i = 0
    for k in range(len(kpi_names)):
        for j in range(len(feature_names)):
            plt.figure()
            plt.plot(features[j], kpis_variable[k][j], 'go', label='data')
            for Model, config in zip(Models, configs):
                plt.plot(features[j], Model(np.asarray(features[j]), *parameters[i]), config, label= Model.name + r': $RMSE=%5.3f$' % rmses[i])
                i += 1
            plt.title(kpi_names[k] + ' vs ' + feature_names[j])
            plt.xlabel(feature_names[j] + ' (' + feature_symbol[j] + ')')
            plt.ylabel(kpi_names[k] + ' (' + kpi_units[k] + ')')
            plt.grid()
            plt.legend()   
    

# ----------------- Strong Regressor System Identification ----------------------------

plt.show()

