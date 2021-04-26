import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import uniform, norm, skewnorm, chi2, ncx2
from scipy.stats import kurtosis, skew
import pickle

# Interactive plotting for matplotlib
plt.ioff()

# Argument configuration
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_plot", type = int, choices=[0, 1],
		 help = "Shows Dataset subsampling used for the System Identification",
		 default = 0)
parser.add_argument("-r", "--result_plot", type = int, choices=[0, 1],
		 help = "Shows resulting plots from the best selected models",
		 default = 0)
parser.add_argument("-v", "--validation_plot", type = int, choices=[0, 1],
		 help = "Shows parameter box plots from 10-fold cross validation ",
		 default = 0)
args = parser.parse_args()

# Load previously generated dataset from DataGenMultivariate.py
file = open('datasetMultivariateCPU.pkl', 'rb')
if not file:
    sys.exit("No datasetMultivariateCPU.pkl file was found")
else:
    dataset = pickle.load(file)
if not dataset:
    sys.exit("Data loaded was empty from datasetMultivariateCPU.pkl file")

# Single Variable Weak Regressor/Learners definition
# Function attributes
class Name:
    def __init__(self, r):
        self.name = r

    def __call__(self, f):
        f.name = self.name
        return f

# Weak regressor with for single-feature modeling before selection
# x: a vector of K samples of a single-feature
# ai: number of parameters per model
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
    

# Error and Loss Metrics function definition
# Mean Absolute Percentage Error (MAPE)
def MAPE(kpi, feature, Model, parameter):
    return np.absolute(np.sum((kpi - Model(np.asarray(feature), *parameter))/kpi))/len(kpi)

# Root Mean Square Error (RMSE)
def RMSE(kpi, feature, Model, parameter):
    return np.sqrt(np.sum((kpi - Model(np.asarray(feature), *parameter)) ** 2)/len(kpi))

# Normalized Root Mean Square Error (NRMSE)
def NRMSE(kpi, feature, Model, parameter):
    return np.sqrt(np.sum((kpi - Model(np.asarray(feature), *parameter)) ** 2)/len(kpi))/(np.max(kpi)-np.min(kpi))

# Total Least Square Error cost function with L1 regularization term   
def LSEL1Cost(kpi, feature, Model, parameter, regLambda):
    return np.sum((kpi - Model(np.asarray(feature), *parameter)) ** 2) + (regLambda * np.sum(np.absolute(parameter)))

# Total Least Square Error cost function with L2 regularization term   
def LSEL2Cost(kpi, feature, Model, parameter, regLambda):
    return np.sum((kpi - Model(np.asarray(feature), *parameter)) ** 2) + (regLambda * np.sum(parameter ** 2))

# MAPE or RMSE cost function with L1 regularization term   
def L1Cost(metric, parameter, regLambda):
    return metric + (regLambda * np.sum(np.absolute(parameter)))

# MAPE or RMSE cost function with L2 regularization term   
def L2Cost(metric, parameter, regLambda):
    return metric + (regLambda * np.sum(parameter ** 2))
    
# Autocorrelation function
def autocorr(x):
    corr = np.correlate(x, x, mode='full')
    return corr   

# ----------------- Weak Regressor System Identification ----------------------------
# Max values of feature space
k_max = 11
N_max = 512
WH_max = 100
C_max = 512
# Defining constant and variable inputs for Dataset subsampling for 3D visualization
# KPIs vs WH and C (filter size k and number of filters N are constant)
k_const = 11
N_const = 512
WH_var, C_var, LAT_WHC, POW_WHC, E_WHC, T_WHC = [],[],[],[],[],[]
# KPIs vs k and N (Input tensor size WH_in and C_in are constant)
WH_const = 100
C_const = 512
k_var, N_var, LAT_kN, POW_kN, E_kN, T_kN = [],[],[],[],[],[]
# Ordered KPI names
kpi_names = ['Latency', 'Power', 'Energy', 'Throughput']
# with units
kpi_units = ['s', 'W', 'J', 'GB/s']
# Ordered feature names
feature_names = ['Input Tensor Size', 'Input Tensor Depth', 'Kernel Size', 'Number of Kernel Filters']
# with 
feature_symbol = ['HW', 'C', 'k', 'N']

# Retrieving sample data from Dataset (sample = [WH, C, k, N, LAT, POW, E, T])
for sample in dataset:
    if sample[2] == k_const and sample[3] == N_const:
        WH_var.append(sample[0])
        C_var.append(sample[1])
        LAT_WHC.append(sample[4]/1000) # in seconds
        POW_WHC.append(sample[5])
        E_WHC.append(sample[6])
        T_WHC.append(sample[7])
    if sample[0] == WH_const and sample[1] == C_const:
        k_var.append(sample[2])
        N_var.append(sample[3])
        LAT_kN.append(sample[4]/1000) # in seconds
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
    ax1.tricontour(X, Y, Z, zdir='x', offset=WH_max, cmap=cm.coolwarm)
    ax1.tricontour(X, Y, Z, zdir='y', offset=C_max, cmap=cm.coolwarm)
    plt.title('Latency vs Input Tensor Size')
    ax1.set_xlabel('Width and Height (WH)')
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
    ax2.tricontour(X, Y, Z, zdir='x', offset=k_max, cmap=cm.coolwarm)
    ax2.tricontour(X, Y, Z, zdir='y', offset=N_max, cmap=cm.coolwarm)
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
    ax3.tricontour(X, Y, Z, zdir='x', offset=WH_max, cmap=cm.coolwarm)
    ax3.tricontour(X, Y, Z, zdir='y', offset=C_max, cmap=cm.coolwarm)
    plt.title('Energy vs Input Tensor Size')
    ax3.set_xlabel('Width and Height (WH)')
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
    ax4.tricontour(X, Y, Z, zdir='x', offset=k_max, cmap=cm.coolwarm)
    ax4.tricontour(X, Y, Z, zdir='y', offset=N_max, cmap=cm.coolwarm)
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
    ax5.tricontour(X, Y, Z, zdir='x', offset=WH_max, cmap=cm.coolwarm)
    ax5.tricontour(X, Y, Z, zdir='y', offset=C_max, cmap=cm.coolwarm)
    plt.title('Power vs Input Tensor Size')
    ax5.set_xlabel('Width and Height (WH)')
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
    ax6.tricontour(X, Y, Z, zdir='x', offset=k_max, cmap=cm.coolwarm)
    ax6.tricontour(X, Y, Z, zdir='y', offset=N_max, cmap=cm.coolwarm)
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
    ax6.tricontour(X, Y, Z, zdir='x', offset=WH_max, cmap=cm.coolwarm)
    ax6.tricontour(X, Y, Z, zdir='y', offset=C_max, cmap=cm.coolwarm)
    plt.title('Throughput vs Input Tensor Size')
    ax6.set_xlabel('Width and Height (WH)')
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
    ax7.tricontour(X, Y, Z, zdir='x', offset=k_max, cmap=cm.coolwarm)
    ax7.tricontour(X, Y, Z, zdir='y', offset=N_max, cmap=cm.coolwarm)
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
        LAT_WH.append(sample[4]/1000)
        POW_WH.append(sample[5])
        E_WH.append(sample[6])
        T_WH.append(sample[7])
    if sample[0] == WH_const and sample[2] == k_const and sample[3] == N_const:
        C_var.append(sample[1])
        LAT_C.append(sample[4]/1000)
        POW_C.append(sample[5])
        E_C.append(sample[6])
        T_C.append(sample[7])
    if sample[0] == WH_const and sample[1] == C_const and sample[3] == N_const:
        k_var.append(sample[2])
        LAT_k.append(sample[4]/1000)
        POW_k.append(sample[5])
        E_k.append(sample[6])
        T_k.append(sample[7])
    if sample[0] == WH_const and sample[1] == C_const and sample[2] == k_const:
        N_var.append(sample[3])
        LAT_N.append(sample[4]/1000)
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
nrmses = []
# Obtained MAPE
mapes = []
# Obtained cost values
costs = []

# Models for each feature by keeping all others features constant using LM method for curve fitting 
for kpis in kpis_variable:
    for feature, kpi in zip(features, kpis):
        for Model in Models:
            parameter, covariance = curve_fit(Model, feature, kpi, maxfev=10000)
            parameters.append(parameter)
            # Computing RMSE
            nrmse = NRMSE(kpi, feature, Model, parameter)
            nrmses.append(nrmse)
            # Computing MAPE
            mape = MAPE(kpi, feature, Model, parameter)
            mapes.append(mape)
            # Computing cost with a LSE metric Loss and L2 regularization
            cost = L2Cost(nrmse, parameter, 10)
            costs.append(cost)               

# ----------------- Strong Regressor System Identification ----------------------------
# Competitive selection by LSE with L2 regularization as Loss function
wr_ensembleCost = []
wr_ensembleParameters = []
selectedModels = []
selectedParameters = []
i = 0
j = 0
k = 0
for kpis in kpis_variable:
    print(kpi_names[i] + ' KPI models:')
    for feature, kpi in zip(features, kpis):
        for Model in Models:
            wr_ensembleCost.append(costs[k])
            wr_ensembleParameters.append(parameters[k])
            k += 1
        selectedModel = Models[np.argmin(wr_ensembleCost)]
        selectedParameter = wr_ensembleParameters[np.argmin(wr_ensembleCost)]
        selectedCost = wr_ensembleCost[np.argmin(wr_ensembleCost)]
        print(selectedModel.name +' model for ' + feature_symbol[j] + ' feature with Cost = %5.3f' % selectedCost + ' with Parameters: ', end = ' ')
        print(selectedParameter)
        selectedModels.append(selectedModel)
        selectedParameters.append(selectedParameter)
        wr_ensembleCost = []
        wr_ensembleParameters = []
        j += 1
    j = 0
    i += 1

# Plot results
if args.result_plot:
    # plot colors and markers configurations
    configs = ['r-', 'cs-', 'm^-', 'bD-', 'yp-']
    k = 0
    for i in range(len(kpi_names)):
        for j in range(len(feature_names)):
            plt.figure()
            plt.plot(features[j], kpis_variable[i][j], 'go', label='data')
            for Model, config in zip(Models, configs):
                plt.plot(features[j], Model(np.asarray(features[j]), *parameters[k]), config, label= Model.name + r': $NRMSE=%5.3f$' % nrmses[k] +  r', $Cost=%5.3f$' % costs[k])
                k += 1
            plt.title(kpi_names[i] + ' vs ' + feature_names[j])
            plt.xlabel(feature_names[j] + ' (' + feature_symbol[j] + ')')
            plt.ylabel(kpi_names[i] + ' (' + kpi_units[i] + ')')
            plt.grid()
            plt.legend()
            

# Multivariate KPI sample initialization
WH, C, K, N = [],[],[],[]
LAT, POW, E, T = [],[],[],[]

# Retrieving KPI samples 
for sample in dataset:
    WH.append(sample[0])
    C.append(sample[1])
    K.append(sample[2])
    N.append(sample[3])
    LAT.append(sample[4]/1000)
    POW.append(sample[5])
    E.append(sample[6])
    T.append(sample[7])

# Reshaping data                
featureData = np.array([WH, C, K, N])

# Strong regressor aggregation or combination for multi-feature modeling after selection
# x: a vector of K samples containing multiple-variables per sample x = (WH, C, k, N) 
# bj: parameters per model. Must be of the same size as x
@Name('Latency Aggregation')
def LatAggModel(x ,b3 ,b2, b1, b0):
    return b0*selectedModels[0](x[0], *selectedParameters[0]) + \
    b1*selectedModels[1](x[1], *selectedParameters[1]) + \
    b2*selectedModels[2](x[2], *selectedParameters[2]) + \
    b3*selectedModels[3](x[3], *selectedParameters[3])
    
@Name('Power Aggregation')
def PowAggModel(x ,b3 ,b2, b1, b0):
    return b0*selectedModels[4](x[0], *selectedParameters[4]) + \
    b1*selectedModels[5](x[1], *selectedParameters[5]) + \
    b2*selectedModels[6](x[2], *selectedParameters[6]) + \
    b3*selectedModels[7](x[3], *selectedParameters[7])
    
@Name('Energy Aggregation')
def EneAggModel(x ,b3 ,b2, b1, b0):
    return b0*selectedModels[8](x[0], *selectedParameters[8]) + \
    b1*selectedModels[9](x[1], *selectedParameters[9]) + \
    b2*selectedModels[10](x[2], *selectedParameters[10]) + \
    b3*selectedModels[11](x[3], *selectedParameters[11])
    
@Name('Throughput Aggregation')
def ThrAggModel(x ,b3 ,b2, b1, b0):
    return b0*selectedModels[12](x[0], *selectedParameters[12]) + \
    b1*selectedModels[13](x[1], *selectedParameters[13]) + \
    b2*selectedModels[14](x[2], *selectedParameters[14]) + \
    b3*selectedModels[15](x[3], *selectedParameters[15])

# Full Dataset identification 
LAT_parameters, LAT_covariance = curve_fit(LatAggModel, featureData, LAT, maxfev=1000)
POW_parameters, POW_covariance = curve_fit(PowAggModel, featureData, POW, maxfev=1000)
E_parameters, E_covariance = curve_fit(EneAggModel, featureData, E, maxfev=1000)
T_parameters, T_covariance = curve_fit(ThrAggModel, featureData, T, maxfev=1000)

# Show resulting NRMSE 
print(NRMSE(LAT, featureData, LatAggModel, LAT_parameters))
print(NRMSE(POW, featureData, PowAggModel, POW_parameters))
print(NRMSE(E, featureData, EneAggModel, E_parameters))
print(NRMSE(T, featureData, ThrAggModel, T_parameters))


#----------------------------------- k-Fold Cross Validation ------------------------------------
if args.validation_plot:
    print("Validating results with 10-Fold Cross-Validation...")
    # Number of folds
    k_folds = 10
    # Number of iterations
    iters = 50
    parameterDistLAT = []
    parameterDistE = []
    avLAT_NMRSE = 0
    avPOW_NMRSE = 0
    avE_NMRSE = 0
    avT_NMRSE = 0
    distNMRSE = []

    # Random seed used to debug 
    #np.random.seed(1234567890)
    for iter in range(iters):
        # Dataset shuffle
        shuffledData = np.array([WH, C, K, N, LAT, POW, E, T])
        shuffledData = shuffledData[:, np.random.permutation(shuffledData.shape[1])]
        # Dataset split in k-folds
        foldSize = shuffledData.shape[1] / k_folds
        for i in range(k_folds):
            # Split for train data
            trainData = np.delete(shuffledData, np.arange(i*foldSize,i*foldSize+foldSize,dtype=int), 1)
            # Split for validation data
            validationData = shuffledData[:,np.arange(i*foldSize,i*foldSize+foldSize,dtype=int)]
            # Identification over training Dataset
            LAT_parameters, LAT_covariance = curve_fit(LatAggModel, trainData[:4,:], trainData[4,:], maxfev=1000)
            POW_parameters, POW_covariance = curve_fit(PowAggModel, trainData[:4,:], trainData[5,:], maxfev=1000)
            E_parameters, E_covariance = curve_fit(EneAggModel, trainData[:4,:], trainData[6,:], maxfev=1000)
            T_parameters, T_covariance = curve_fit(ThrAggModel, trainData[:4,:], trainData[7,:], maxfev=1000)
            # Compute resulting NRMSE on validation Dataset fold
            distNMRSE.append(NRMSE(validationData[6,:], validationData[:4,:], EneAggModel, E_parameters))
            avLAT_NMRSE += NRMSE(validationData[4,:], validationData[:4,:], LatAggModel, LAT_parameters)
            avPOW_NMRSE += NRMSE(validationData[5,:], validationData[:4,:], PowAggModel, POW_parameters)
            avE_NMRSE += NRMSE(validationData[6,:], validationData[:4,:], EneAggModel, E_parameters)
            avT_NMRSE += NRMSE(validationData[7,:], validationData[:4,:], ThrAggModel, T_parameters)
            # Store obtained distribution per fold iteration
            parameterDistLAT.append(np.concatenate((LAT_parameters[0]*selectedParameters[0], \
                                    LAT_parameters[1]*selectedParameters[1], \
                                    LAT_parameters[2]*selectedParameters[2], \
                                    LAT_parameters[3]*selectedParameters[3])))
            parameterDistE.append(np.concatenate((E_parameters[0]*selectedParameters[8], \
                                    E_parameters[1]*selectedParameters[9], \
                                    E_parameters[2]*selectedParameters[10], \
                                    E_parameters[3]*selectedParameters[11])))
    # Average NRMSE metric
    k_folds = k_folds*iters
    avLAT_NMRSE = avLAT_NMRSE / k_folds
    avPOW_NMRSE = avPOW_NMRSE / k_folds
    avE_NMRSE = avE_NMRSE / k_folds
    avT_NMRSE = avT_NMRSE / k_folds

    #------------------------------------- Gaussian normal distribution hypothesis testing with KS ------------------------------
    distLATarray = np.array(parameterDistLAT)
    distEarray = np.array(parameterDistE)
    # Normalise
    distLATarray = (distLATarray - np.mean(distLATarray, axis=0)) / (np.amax(distLATarray, axis=0) - np.amin(distLATarray, axis=0))
    distEarray = (distEarray - np.mean(distEarray, axis=0))/ (np.amax(distEarray, axis=0) - np.amin(distEarray, axis=0))
    # Standarise
    #distLATarray = (distLATarray - np.mean(distLATarray, axis=0)) / np.std(distLATarray, axis=0)
    #distEarray = (distEarray - np.mean(distEarray, axis=0)) / np.std(distEarray, axis=0)

    # Distribution Test
    y = np.array(distNMRSE)
    y_centered = (y - np.mean(y)) / (np.max(y) - np.min(y))
    y = (y-np.mean(y))/np.std(y)
    plt.figure()
    plt.plot(y, label='NRMSE')
    plt.title('NRMSE per k-fold validation iteration')
    plt.grid()
    x = np.linspace(np.min(y),np.max(y),iters)
    bins = np.linspace(np.min(y),np.max(y),iters+1)
    hist, bin_edges = np.histogram(y, bins=bins, density=True)
    mean, var = norm.fit(y) # Get first 2 moments of data
    smean, svar, sk = skewnorm.fit(y) # Get first 3 moments of data
    #df, nc, cmean, cvar = ncx2.fit(y, 4) # Get 3 first moments of data
    gNorm = norm.pdf(x, mean, var) # Center and scale a Gaussian function
    sNorm = skewnorm.pdf(x, smean, svar, sk) # Center and scale a Skewed Gaussian function
    #chiSq = ncx2.pdf(x, 4, 1) # Center and scale a Chi-Square function
    plt.figure()
    plt.plot(x, gNorm,'r-', label='Norm PDF')
    plt.plot(x, sNorm,'g-', label='Skewed Norm PDF')
    #plt.plot(x, chiSq,'m-', label='Chi-Square PDF')
    plt.bar(bin_edges[:-1], hist, width = (max(bin_edges)-min(bin_edges))/iters)
    plt.title('NRMSE distribution')
    plt.xlim(np.min(x),np.max(x)) 
    plt.legend()
    # Print Normality Test results (Variable and p-value)
    #   Kolmogorov-Smirnov
    print(stats.kstest(y, 'norm', args=(mean, var)))
    print(stats.kstest(y, 'skewnorm', args=(smean, svar, sk)))
    #   Shapiro-Wilk
    print(stats.shapiro(y))
    #   Chi-Square
    #print(stats.chisquare(hist, gNorm))
    #print(stats.chisquare(hist, sNorm))
    # Q-Q plots
    #f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    f, (ax1,ax2) = plt.subplots(1,2)
    plt.title("Q-Q Plots")
    res = stats.probplot(y, dist=stats.norm(mean, var), plot=ax1)
    ax1.set_title("Normality Test (Non-Skewed)")
    resS = stats.probplot(y, dist=stats.skewnorm(smean, svar, sk), plot=ax2)
    ax2.set_title("Normality Test (Skewed)")
    #resX2 = stats.probplot(y, dist=stats.chi2(4), plot=ax3)
    #ax3.set_title("Chi-Square Test (k=4)")
    #resX2 = stats.probplot(y, dist=stats.chi2(10), plot=ax4)
    #ax4.set_title("Chi-Square Test (k=10)")
    # Auto-correlation analysys for White- or Gaussian-noise 
    y = y_centered
    corr = autocorr(y)
    plt.figure()
    plt.bar(np.linspace(0,k_folds-1,k_folds),corr[k_folds-1:])
    plt.title('NRMSE Auto-correlation')
    plt.grid()
    print('Average zero-element correlation value: %5.3f should be almost equal to variance: %5.3f' % (corr[k_folds-1]/k_folds, np.std(y)**2))
    print('Average of non-zero correlation values: %5.3f should be almost equal to 0' % np.mean(corr[k_folds:]))
    # Show parameter distribution after k-fold validation for box plotting    

    plt.figure()
    plt.violinplot(distLATarray,showmeans=False,showmedians=True,showextrema=True)
    plt.title('Parameter distribution for Latency model')
    plt.xlabel('Parameters')
    plt.figure()
    plt.boxplot(distLATarray, 0, '')
    plt.title('Parameter distribution for Latency model')
    plt.xlabel('Parameters')
    plt.xticks(np.arange(13), ('' , 'a11', 'a10', 'a9', 'a8','a7', 'a6', 'a5', 'a4', 'a3','a2', 'a1', 'a0'))
    plt.grid()
    plt.figure()
    plt.violinplot(distEarray,showmeans=False,showmedians=True,showextrema=True)
    plt.title('Parameter distribution for Energy model')
    plt.xlabel('Parameters')
    plt.figure()
    plt.boxplot(distEarray, 0, '')
    plt.title('Parameter distribution for Energy model')
    plt.xlabel('Parameters')
    plt.xticks(np.arange(15), ('', 'a13', 'a12' , 'a11', 'a10', 'a9', 'a8','a7', 'a6', 'a5', 'a4', 'a3','a2', 'a1', 'a0'))
    plt.grid()
    #for i in range(distLATarray.shape[1]):
        #plt.figure()
        #plt.boxplot(np.transpose(distLATarray)[i])
        
    
plt.show()
#plt.close('all')


