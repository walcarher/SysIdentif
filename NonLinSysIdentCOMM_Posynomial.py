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

# Latency dataset in ms with Input data in KB
dataset = np.array([[2,4,8,16,32,64,128,256,512,1024],
                   [0.013064032,0.013998889,0.015018493,0.017993855,0.027997778,0.048984877,0.091029316,0.176020638,0.347037136,0.699042591],
                   [0.021192763,0.021999408,0.021993066,0.021993066,0.027997778,0.047983613,0.08602559,0.164959882,0.321026463,0.651910881]])

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

# Defining constant and variable inputs for Dataset 2D visualization
# KPIs vs datasize (WxHxC for IFM and (H-k+1)x(W-k+1)xN for OFM)
# Ordered KPI names
kpi_names = ['Latency']
# with units
kpi_units = ['ms']
# Ordered feature names
feature_names = ['FM Data Size']
# with symbols
feature_symbol = ['KB']

# Plot subsampled dataset for 3D Visualization if data_plot is enabled
if args.data_plot:
    for kpi_name, kpi_unit in zip(kpi_names, kpi_units):
        # KPI vs Input tensor data size
        fig = plt.figure()
        X = np.array(dataset[0])
        Y1 = np.array(dataset[1])
        Y2 = np.array(dataset[2])
        plt.plot(X, Y1, 'go', label='IFM data')
        plt.plot(X, Y2, 'ro', label='OFM data')
        plt.title(kpi_name + ' vs ' + feature_names[0])
        plt.xlabel(feature_names[0] + ' (' + feature_symbol[0] + ')')
        plt.ylabel(kpi_name + ' (' + kpi_unit + ')')
        plt.grid()
        plt.legend()
        
# Lists of KPIs per feature
LAT_IFM_data = dataset[1]
LAT_OFM_data = dataset[2]
kpis_variable = [[LAT_IFM_data, LAT_OFM_data]]
# List of features
features = [dataset[0]]
# Previously defined models
Models = [LinModel, QuadModel, LogModel, ExpModel, ReciModel, PolyModel]
#Models = [LinModel, QuadModel, ExpModel, ReciModel, PolyModel]
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


# Full Dataset identification 
# LAT_parameters, LAT_covariance = curve_fit(LatAggModel,
                                           # featureData, 
                                           # LAT, 
                                           # p0=np.concatenate(selectedParameters[0:4]), 
                                           # bounds=[np.zeros(LatAggModel.parameter_number), np.inf*np.ones(LatAggModel.parameter_number)], 
                                           # maxfev=1000)

# # Print Strong regressor parameters
# print('Strong regressor parameters:')
# print('Latency parameters: ' + np.array2string(LAT_parameters))

# # Show resulting NRMSE 
# print('Precision metrics:')
# print('Latency NRMSE: ' + str(NRMSE(LAT, featureData, LatAggModel, LAT_parameters)))

# fileLAT = open('parametersLATCOMM.pkl', 'wb')
# pickle.dump(LAT_parameters, fileLAT)
                
 
plt.show()
#plt.close('all')


