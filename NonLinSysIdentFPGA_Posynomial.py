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
parser.add_argument("-m", "--model_plot", type = int, choices=[0, 1],
		 help = "Shows resulting plots from the strong regressor model",
		 default = 0)
parser.add_argument("-v", "--validation_plot", type = int, choices=[0, 1],
		 help = "Shows parameter box plots from 10-fold cross validation ",
		 default = 0)
args = parser.parse_args()

# Load previously generated dataset from DataGenMultivariateFPGA.py
file = open('datasetMultivariateFPGA.pkl', 'rb')
if not file:
    sys.exit("No datasetMultivariateFPGA.pkl file was found")
else:
    dataset = pickle.load(file)
if not dataset:
    sys.exit("Data loaded was empty from datasetMultivariateFPGA.pkl file")

# Single Variable Weak Regressor/Learners definition
# Function attributes
class Name:
    def __init__(self, r):
        self.name = r

    def __call__(self, f):
        f.name = self.name
        return f
        
class ParameterNumber:
    def __init__(self, r):
        self.parameter_number = r

    def __call__(self, f):
        f.parameter_number = self.parameter_number
        return f

# Weak regressor with for single-feature modeling before selection
# x: a vector of K samples of a single-feature
# ai: number of parameters per model
# Linear model
@Name('Linear')
@ParameterNumber(2)
def LinModel(x, a1, a0):
    return a1*x + a0

# Quadratic model
@Name('Quadratic')
@ParameterNumber(3)
def QuadModel(x, a2, a1, a0):
    return a2*np.power(x, 2) + a1*x + a0

# Logarithmic model
@Name('Logarithmic')  
@ParameterNumber(2)
def LogModel(x, a1, a0):
    return a1*np.log(x) + a0
    
# Exponential model   
@Name('Exponential') 
@ParameterNumber(3)
def ExpModel(x, a2, a1, a0):
    return a2*np.power(a1, x) + a0
    
# Reciprocal model   
@Name('Reciprocal') 
@ParameterNumber(2)
def ReciModel(x, a1, a0):
    return a1*(1/x) + a0
    
# Polynomial model   
@Name('Polynomial') 
@ParameterNumber(4)
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
k_max = 5
N_max = 10
WH_max = 12
C_max = 10
# Defining constant and variable inputs for Dataset subsampling for 3D visualization
# KPIs vs WH and C (filter size k and number of filters N are constant)
k_const = 3
N_const = 5
WH_var, C_var, LAT_WHC, POW_WHC, E_WHC, T_WHC = [],[],[],[],[],[]
ALM_WHC, ALUT_WHC, LAB_WHC, M20K_WHC = [],[],[],[]
# KPIs vs k and N (Input tensor size WH_in and C_in are constant)
WH_const = 5
C_const = 5
k_var, N_var, LAT_kN, POW_kN, E_kN, T_kN = [],[],[],[],[],[]
ALM_kN, ALUT_kN, LAB_kN, M20K_kN = [],[],[],[]
# Ordered KPI names
kpi_names = ['Latency', 'Power', 'Energy', 'Throughput', 'ALMs', 'ALUTs', 'LABs', 'M20Ks']
# with units
kpi_units = ['ms', 'W', r'$\mu$J', 'GB/s', '#', '#', '#', '#']
# Ordered feature names
feature_names = ['Input Tensor Size', 'Input Tensor Depth', 'Kernel Size', 'Number of Kernel Filters']
# with symbols
feature_symbol = ['HW', 'C', 'k', 'N']

# Retrieving sample data from Dataset (sample = [WH, C, k, N, LAT, POW, E, T, R_ALM, R_ALUT, R_LAB, R_M20K])
for sample in dataset:
    if sample[2] == k_const and sample[3] == N_const:
        WH_var.append(sample[0])
        C_var.append(sample[1])
        LAT_WHC.append(sample[4])
        POW_WHC.append(sample[5])
        E_WHC.append(sample[6]*1000000) # in microJoules
        T_WHC.append(sample[7])
        ALM_WHC.append(sample[8])
        ALUT_WHC.append(sample[9])
        LAB_WHC.append(sample[10])
        M20K_WHC.append(sample[11])
    if sample[0] == WH_const and sample[1] == C_const:
        k_var.append(sample[2])
        N_var.append(sample[3])
        LAT_kN.append(sample[4])
        POW_kN.append(sample[5])
        E_kN.append(sample[6]*1000000) # in microJoules
        T_kN.append(sample[7])
        ALM_kN.append(sample[8])
        ALUT_kN.append(sample[9])
        LAB_kN.append(sample[10])
        M20K_kN.append(sample[11])

kpi_WHC = [LAT_WHC, POW_WHC, E_WHC, T_WHC, ALM_WHC, ALUT_WHC, LAB_WHC, M20K_WHC]        
kpi_kN = [LAT_kN, POW_kN, E_kN, T_kN, ALM_kN, ALUT_kN, LAB_kN, M20K_kN]
# Plot subsampled dataset for 3D Visualization if data_plot is enabled
if args.data_plot:
    for kpi_whc, kpi_kn, kpi_name, kpi_unit in zip(kpi_WHC, kpi_kN, kpi_names, kpi_units):
        # KPI vs Input tensor size (KPI vs WH and C) with kernel size and depth constant (k and N)
        fig = plt.figure()
        X = np.array(WH_var)
        Y = np.array(C_var)
        Z = np.array(kpi_whc)
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
        ax.tricontour(X, Y, Z, zdir='x', offset=WH_max, cmap=cm.coolwarm)
        ax.tricontour(X, Y, Z, zdir='y', offset=C_max, cmap=cm.coolwarm)
        plt.title(kpi_name + ' vs ' + feature_names[0])
        ax.set_xlabel(feature_names[0] + ' (' + feature_symbol[0] + ')')
        #ax.set_xlim(,)
        ax.set_ylabel(feature_names[1] + ' (' + feature_symbol[1] + ')')
        #ax.set_ylim(,)
        ax.set_zlabel(kpi_name + ' (' + kpi_unit + ')')
        #ax.set_zlim(,)
        # KPI vs Filter size and depth (KPI vs k and N) with input tensor size constant (WH and C)
        fig = plt.figure()
        X = np.array(k_var)
        Y = np.array(N_var)
        Z = np.array(kpi_kn)
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm, alpha=0.3)
        ax.tricontour(X, Y, Z, zdir='x', offset=k_max, cmap=cm.coolwarm)
        ax.tricontour(X, Y, Z, zdir='y', offset=N_max, cmap=cm.coolwarm)
        plt.title(kpi_name + ' vs ' + feature_names[2])
        ax.set_xlabel(feature_names[2] + ' (' + feature_symbol[2] + ')')
        #ax.set_xlim(,)
        ax.set_ylabel(feature_names[3] + ' (' + feature_symbol[3] + ')')
        #ax.set_ylim(,)
        ax.set_zlabel(kpi_name + ' (' + kpi_unit + ')')
        #ax.set_zlim(,)


# Defining constant and variable inputs for Dataset subsampling for SI
WH_var, LAT_WH, POW_WH, E_WH, T_WH = [],[],[],[],[]
ALM_WH, ALUT_WH, LAB_WH, M20K_WH = [],[],[],[]
C_var, LAT_C, POW_C, E_C, T_C = [],[],[],[],[]
ALM_C, ALUT_C, LAB_C, M20K_C = [],[],[],[]
k_var, LAT_k, POW_k, E_k, T_k = [],[],[],[],[]
ALM_k, ALUT_k, LAB_k, M20K_k = [],[],[],[]
N_var, LAT_N, POW_N, E_N, T_N = [],[],[],[],[]
ALM_N, ALUT_N, LAB_N, M20K_N = [],[],[],[]


# Retrieving sample data from Dataset for SI(sample = [WH, C, k, N, LAT, POW, E, T, R_ALM, R_ALUT, R_LAB, R_M20K])
for sample in dataset:
    if  sample[1] == C_const and sample[2] == k_const and sample[3] == N_const:
        WH_var.append(sample[0])
        LAT_WH.append(sample[4])
        POW_WH.append(sample[5])
        E_WH.append(sample[6]*1000000)
        T_WH.append(sample[7])
        ALM_WH.append(sample[8])
        ALUT_WH.append(sample[9])
        LAB_WH.append(sample[10])
        M20K_WH.append(sample[11])
    if sample[0] == WH_const and sample[2] == k_const and sample[3] == N_const:
        C_var.append(sample[1])
        LAT_C.append(sample[4])
        POW_C.append(sample[5])
        E_C.append(sample[6]*1000000)
        T_C.append(sample[7])
        ALM_C.append(sample[8])
        ALUT_C.append(sample[9])
        LAB_C.append(sample[10])
        M20K_C.append(sample[11])
    if sample[0] == WH_const and sample[1] == C_const and sample[3] == N_const:
        k_var.append(sample[2])
        LAT_k.append(sample[4])
        POW_k.append(sample[5])
        E_k.append(sample[6]*1000000)
        T_k.append(sample[7])
        ALM_k.append(sample[8])
        ALUT_k.append(sample[9])
        LAB_k.append(sample[10])
        M20K_k.append(sample[11])
    if sample[0] == WH_const and sample[1] == C_const and sample[2] == k_const:
        N_var.append(sample[3])
        LAT_N.append(sample[4])
        POW_N.append(sample[5])
        E_N.append(sample[6]*1000000)
        T_N.append(sample[7])
        ALM_N.append(sample[8])
        ALUT_N.append(sample[9])
        LAB_N.append(sample[10])
        M20K_N.append(sample[11])
        
        
# Lists of KPIs per feature
kpis_variable = [[LAT_WH, LAT_C, LAT_k, LAT_N], [POW_WH, POW_C, POW_k, POW_N], [E_WH, E_C, E_k, E_N], [T_WH, T_C, T_k, T_N], \
                [ALM_WH, ALM_C, ALM_k, ALM_N], [ALUT_WH, ALUT_C, ALUT_k, ALUT_N], [LAB_WH, LAB_C, LAB_k, LAB_N], [M20K_WH, M20K_C, M20K_k, M20K_N]]
# List of features
features = [WH_var, C_var, k_var, N_var]
# Previously defined models
#Models = [LinModel, QuadModel, LogModel, ExpModel, ReciModel, PolyModel]
Models = [LinModel, QuadModel, ExpModel, ReciModel, PolyModel]
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
            parameter, covariance = curve_fit(Model,
                                              feature,
                                              kpi,
                                              bounds=[np.zeros(Model.parameter_number), np.inf*np.ones(Model.parameter_number)],
                                              maxfev=1000)
            parameters.append(parameter)
            # Computing RMSE
            nrmse = RMSE(kpi, feature, Model, parameter)
            nrmses.append(nrmse)
            # Computing MAPE
            mape = MAPE(kpi, feature, Model, parameter)
            mapes.append(mape)
            # Computing cost with a LSE metric Loss and L2 regularization
            cost = L2Cost(nrmse, parameter, 1000)
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
    configs = ['r-', 'cs-', 'm^-', 'bD-', 'yp-', 'k*-']
    k = 0
    for i in range(len(kpi_names)):
        for j in range(len(feature_names)):
            plt.figure()
            plt.plot(features[j], kpis_variable[i][j], 'go', label='data')
            for Model, config in zip(Models, configs):
                order = np.argsort(features[j])
                plt.plot(np.asarray(features[j])[order], Model(np.asarray(features[j])[order], *parameters[k]), config, label= Model.name + r': $NRMSE=%5.3f$' % nrmses[k] +  r', $Cost=%5.3f$' % costs[k])
                k += 1
            plt.title(kpi_names[i] + ' vs ' + feature_names[j])
            plt.xlabel(feature_names[j] + ' (' + feature_symbol[j] + ')')
            plt.ylabel(kpi_names[i] + ' (' + kpi_units[i] + ')')
            plt.grid()
            plt.legend()
            

# Multivariate KPI sample initialization
WH, C, K, N = [],[],[],[]
LAT, POW, E, T = [],[],[],[]
ALM, ALUT, LAB, M20K = [],[],[],[]

# Retrieving KPI samples 
for sample in dataset:
    WH.append(sample[0])
    C.append(sample[1])
    K.append(sample[2])
    N.append(sample[3])
    LAT.append(sample[4])
    POW.append(sample[5])
    E.append(sample[6]*1000000)
    T.append(sample[7])
    ALM.append(sample[8])
    ALUT.append(sample[9])
    LAB.append(sample[10])
    M20K.append(sample[11])

# Reshaping data                
featureData = np.array([WH, C, K, N])

# Strong regressor aggregation or combination for multi-feature modeling after selection
# x: a vector of K samples containing multiple-variables per sample x = (WH, C, k, N) 
# b: parameters per model. Must be of the same size as x
@Name('Latency Aggregation')
@ParameterNumber(selectedModels[0].parameter_number+selectedModels[1].parameter_number + \
                 selectedModels[2].parameter_number+selectedModels[3].parameter_number)
def LatAggModel(x, *b):
    index = len(selectedParameters[0])
    HW_model = selectedModels[0](x[0], *b[0:index])
    index2 = index + len(selectedParameters[1])
    C_model = selectedModels[1](x[1], *b[index:index2])
    index3 = index2 + len(selectedParameters[2])
    k_model = selectedModels[2](x[2], *b[index2:index3])
    index4 = index3 + len(selectedParameters[3])
    N_model = selectedModels[3](x[3], *b[index3:index4])
    return HW_model * C_model * k_model * N_model
    
@Name('Power Aggregation')
@ParameterNumber(selectedModels[4].parameter_number+selectedModels[5].parameter_number + \
                 selectedModels[6].parameter_number+selectedModels[7].parameter_number)
def PowAggModel(x ,*b):
    index = len(selectedParameters[4])
    HW_model = selectedModels[4](x[0], *b[0:index])
    index2 = index + len(selectedParameters[5])
    C_model = selectedModels[5](x[1], *b[index:index2])
    index3 = index2 + len(selectedParameters[6])
    k_model = selectedModels[6](x[2], *b[index2:index3])
    index4 = index3 + len(selectedParameters[7])
    N_model = selectedModels[7](x[3], *b[index3:index4])
    return HW_model * C_model * k_model * N_model
    
@Name('Energy Aggregation')
@ParameterNumber(selectedModels[8].parameter_number+selectedModels[9].parameter_number + \
                 selectedModels[10].parameter_number+selectedModels[11].parameter_number)
def EneAggModel(x ,*b):
    index = len(selectedParameters[8])
    HW_model = selectedModels[8](x[0], *b[0:index])
    index2 = index + len(selectedParameters[9])
    C_model = selectedModels[9](x[1], *b[index:index2])
    index3 = index2 + len(selectedParameters[10])
    k_model = selectedModels[10](x[2], *b[index2:index3])
    index4 = index3 + len(selectedParameters[11])
    N_model = selectedModels[11](x[3], *b[index3:index4])
    return HW_model * C_model * k_model * N_model
    
@Name('Throughput Aggregation')
@ParameterNumber(selectedModels[12].parameter_number+selectedModels[13].parameter_number + \
                 selectedModels[14].parameter_number+selectedModels[15].parameter_number)
def ThrAggModel(x ,*b):
    index = len(selectedParameters[12])
    HW_model = selectedModels[12](x[0], *b[0:index])
    index2 = index + len(selectedParameters[13])
    C_model = selectedModels[13](x[1], *b[index:index2])
    index3 = index2 + len(selectedParameters[14])
    k_model = selectedModels[14](x[2], *b[index2:index3])
    index4 = index3 + len(selectedParameters[15])
    N_model = selectedModels[15](x[3], *b[index3:index4])
    return HW_model * C_model * k_model * N_model
    
@Name('ALM Aggregation')
@ParameterNumber(selectedModels[16].parameter_number+selectedModels[17].parameter_number + \
                 selectedModels[18].parameter_number+selectedModels[19].parameter_number)
def ALMAggModel(x ,*b):
    index = len(selectedParameters[16])
    HW_model = selectedModels[16](x[0], *b[0:index])
    index2 = index + len(selectedParameters[17])
    C_model = selectedModels[17](x[1], *b[index:index2])
    index3 = index2 + len(selectedParameters[18])
    k_model = selectedModels[18](x[2], *b[index2:index3])
    index4 = index3 + len(selectedParameters[19])
    N_model = selectedModels[19](x[3], *b[index3:index4])
    return HW_model * C_model * k_model * N_model

@Name('ALUT Aggregation')
@ParameterNumber(selectedModels[20].parameter_number+selectedModels[21].parameter_number + \
                 selectedModels[22].parameter_number+selectedModels[23].parameter_number)
def ALUTAggModel(x ,*b):
    index = len(selectedParameters[20])
    HW_model = selectedModels[20](x[0], *b[0:index])
    index2 = index + len(selectedParameters[21])
    C_model = selectedModels[21](x[1], *b[index:index2])
    index3 = index2 + len(selectedParameters[22])
    k_model = selectedModels[22](x[2], *b[index2:index3])
    index4 = index3 + len(selectedParameters[23])
    N_model = selectedModels[23](x[3], *b[index3:index4])
    return HW_model * C_model * k_model * N_model

@Name('LAB Aggregation')
@ParameterNumber(selectedModels[24].parameter_number+selectedModels[25].parameter_number + \
                 selectedModels[26].parameter_number+selectedModels[27].parameter_number)
def LABAggModel(x ,*b):
    index = len(selectedParameters[24])
    HW_model = selectedModels[24](x[0], *b[0:index])
    index2 = index + len(selectedParameters[25])
    C_model = selectedModels[25](x[1], *b[index:index2])
    index3 = index2 + len(selectedParameters[26])
    k_model = selectedModels[26](x[2], *b[index2:index3])
    index4 = index3 + len(selectedParameters[27])
    N_model = selectedModels[27](x[3], *b[index3:index4])
    return HW_model * C_model * k_model * N_model
    
@Name('M20K Aggregation')
@ParameterNumber(selectedModels[28].parameter_number+selectedModels[29].parameter_number + \
                 selectedModels[30].parameter_number+selectedModels[31].parameter_number)
def M20KAggModel(x ,*b):
    index = len(selectedParameters[28])
    HW_model = selectedModels[28](x[0], *b[0:index])
    index2 = index + len(selectedParameters[29])
    C_model = selectedModels[29](x[1], *b[index:index2])
    index3 = index2 + len(selectedParameters[30])
    k_model = selectedModels[30](x[2], *b[index2:index3])
    index4 = index3 + len(selectedParameters[31])
    N_model = selectedModels[31](x[3], *b[index3:index4])
    return HW_model * C_model * k_model * N_model

# Full Dataset identification 
LAT_parameters, LAT_covariance = curve_fit(LatAggModel,
                                           featureData, 
                                           LAT, 
                                           p0=np.concatenate(selectedParameters[0:4]), 
                                           bounds=[np.zeros(LatAggModel.parameter_number), np.inf*np.ones(LatAggModel.parameter_number)], 
                                           maxfev=1000)
POW_parameters, POW_covariance = curve_fit(PowAggModel,
                                           featureData, 
                                           POW, 
                                           p0=np.concatenate(selectedParameters[4:8]),
                                           bounds=[np.zeros(PowAggModel.parameter_number), np.inf*np.ones(PowAggModel.parameter_number)],
                                           maxfev=1000)
E_parameters, E_covariance = curve_fit(EneAggModel, 
                                       featureData, 
                                       E, 
                                       p0=np.concatenate(selectedParameters[8:12]),
                                       bounds=[np.zeros(EneAggModel.parameter_number), np.inf*np.ones(EneAggModel.parameter_number)],
                                       maxfev=1000)
T_parameters, T_covariance = curve_fit(ThrAggModel, 
                                       featureData, 
                                       T, 
                                       p0=np.concatenate(selectedParameters[12:16]),
                                       bounds=[np.zeros(ThrAggModel.parameter_number), np.inf*np.ones(ThrAggModel.parameter_number)],
                                       maxfev=1000)
ALM_parameters, ALM_covariance = curve_fit(ALMAggModel,
                                           featureData, 
                                           ALM, 
                                           p0=np.concatenate(selectedParameters[16:20]), 
                                           bounds=[np.zeros(ALMAggModel.parameter_number), np.inf*np.ones(ALMAggModel.parameter_number)],
                                           maxfev=1000)
ALUT_parameters, ALUT_covariance = curve_fit(ALUTAggModel, 
                                             featureData, 
                                             ALUT, 
                                             p0=np.concatenate(selectedParameters[20:24]),
                                             bounds=[np.zeros(ALUTAggModel.parameter_number), np.inf*np.ones(ALUTAggModel.parameter_number)],
                                             maxfev=1000)
LAB_parameters, LAB_covariance = curve_fit(LABAggModel, 
                                           featureData, 
                                           LAB, 
                                           p0=np.concatenate(selectedParameters[24:28]), 
                                           bounds=[np.zeros(LABAggModel.parameter_number), np.inf*np.ones(LABAggModel.parameter_number)],
                                           maxfev=1000)
M20K_parameters, M20K_covariance = curve_fit(M20KAggModel,
                                             featureData, 
                                             M20K, 
                                             p0=np.concatenate(selectedParameters[28:32]), 
                                             bounds=[np.zeros(M20KAggModel.parameter_number), np.inf*np.ones(M20KAggModel.parameter_number)],
                                             maxfev=1000)

# Print Strong regressor parameters
print('Strong regressor parameters:')
print('Latency parameters: ' + np.array2string(LAT_parameters))
print('Power parameters: ' + np.array2string(POW_parameters))
print('Energy parameters: ' + np.array2string(E_parameters))
print('Throughput parameters: ' + np.array2string(T_parameters))
print('ALM parameters: ' + np.array2string(ALM_parameters))
print('ALUT parameters: ' + np.array2string(ALUT_parameters))
print('LAB parameters: ' + np.array2string(LAB_parameters))
print('M20K parameters: ' + np.array2string(M20K_parameters))

# Show resulting NRMSE 
# Show resulting NRMSE 
print('Precision metrics:')
print('Latency NRMSE: ' + str(NRMSE(LAT, featureData, LatAggModel, LAT_parameters)))
print('Power NRMSE: ' + str(NRMSE(POW, featureData, PowAggModel, POW_parameters)))
print('Energy NRMSE: ' + str(NRMSE(E, featureData, EneAggModel, E_parameters)))
print('Throughput NRMSE: ' + str(NRMSE(T, featureData, ThrAggModel, T_parameters)))
print('ALM NRMSE: ' + str(NRMSE(ALM, featureData, ALMAggModel, ALM_parameters)))
print('ALUT NRMSE: ' + str(NRMSE(ALUT, featureData, ALUTAggModel, ALUT_parameters)))
print('LAB NRMSE: ' + str(NRMSE(LAB, featureData, LABAggModel, LAB_parameters)))
print('M20K NRMSE: ' + str(NRMSE(M20K, featureData, M20KAggModel, M20K_parameters)))

fileLAT = open('parametersLATFPGA.pkl', 'wb')
pickle.dump(LAT_parameters, fileLAT)
                
fileE = open('parametersEFPGA.pkl', 'wb')
pickle.dump(E_parameters, fileE)

fileALM = open('parametersALMFPGA.pkl', 'wb')
pickle.dump(ALM_parameters, fileALM)

fileALUT = open('parametersALUTFPGA.pkl', 'wb')
pickle.dump(ALUT_parameters, fileALUT)

fileLAB = open('parametersLABFPGA.pkl', 'wb')
pickle.dump(LAB_parameters, fileLAB)

fileM20K = open('parametersM20KFPGA.pkl', 'wb')
pickle.dump(M20K_parameters, fileM20K)

# Plot models 
if args.model_plot:
    strRegModels = [LatAggModel, PowAggModel, EneAggModel, ThrAggModel, ALMAggModel, ALUTAggModel, LABAggModel, M20KAggModel]
    strParameters = [LAT_parameters, POW_parameters, E_parameters, T_parameters, ALM_parameters, ALUT_parameters, LAB_parameters, M20K_parameters]
    for Model, parameters, kpi_name, kpi_unit in zip(strRegModels, strParameters, kpi_names, kpi_units):
        WH_mod = np.arange(1, 13, 1)
        C_mod = np.arange(1, 11, 1)
        X, Y = np.meshgrid(WH_mod, C_mod)
        k_mod = k_const*np.ones_like(X)
        N_mod = N_const*np.ones_like(Y)
        Z = Model([X,Y,k_mod, N_mod],*parameters)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        plt.title(kpi_name + ' Model vs ' + feature_names[0])
        ax.set_xlabel(feature_names[0] + ' (' + feature_symbol[0] + ')')
        #ax.set_xlim(,)
        ax.set_ylabel(feature_names[1] + ' (' + feature_symbol[1] + ')')
        #ax.set_ylim(,)
        ax.set_zlabel(kpi_name + ' (' + kpi_unit + ')')
                               
        k_mod = np.arange(1, 6, 1)
        N_mod = np.arange(1, 11, 1)
        X, Y = np.meshgrid(k_mod, N_mod)
        WH_mod = WH_const*np.ones_like(X)
        C_mod = C_const*np.ones_like(Y)
        Z = Model([WH_mod,C_mod,X,Y],*parameters)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        plt.title(kpi_name + ' Model vs ' + feature_names[2])
        ax.set_xlabel(feature_names[2] + ' (' + feature_symbol[2] + ')')
        #ax.set_xlim(,)
        ax.set_ylabel(feature_names[3] + ' (' + feature_symbol[3] + ')')
        #ax.set_ylim(,)
        ax.set_zlabel(kpi_name + ' (' + kpi_unit + ')')
        #ax.set_zlim(,)

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
    normdistLATarray = (distLATarray - np.mean(distLATarray, axis=0)) / (np.amax(distLATarray, axis=0) - np.amin(distLATarray, axis=0))
    normdistEarray = (distEarray - np.mean(distEarray, axis=0))/ (np.amax(distEarray, axis=0) - np.amin(distEarray, axis=0))
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
    plt.violinplot(normdistLATarray,showmeans=False,showmedians=True,showextrema=True)
    plt.title('Parameter distribution for Latency model')
    plt.xlabel('Parameters')
    plt.figure()
    plt.boxplot(distLATarray, 0, '')
    plt.title('Parameter distribution for Latency model')
    plt.xlabel('Parameters')
    plt.xticks(np.arange(13), ('' , 'a11', 'a10', 'a9', 'a8','a7', 'a6', 'a5', 'a4', 'a3','a2', 'a1', 'a0'))
    plt.grid()
    plt.figure()
    plt.violinplot(normdistEarray,showmeans=False,showmedians=True,showextrema=True)
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


