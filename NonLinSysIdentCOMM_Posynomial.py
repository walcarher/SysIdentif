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
dataset = np.array([[2,4,8,16,32,64,128,256,512,1024]
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
    # KPI vs Input tensor data size
    fig = plt.figure()
    X = np.array(dataset[0])
    Y1 = np.array(dataset[1])
    Y2 = np.array(dataset[2])
    plt.plot(X, Y1, Y2)
    plt.title(kpi_name + ' vs ' + feature_names[0])
    ax.set_xlabel(feature_names[0] + ' (' + feature_symbol[0] + ')')
    #ax.set_xlim(,)
    ax.set_ylabel(kpi_name + ' (' + kpi_unit + ')')
    #ax.set_ylim(,)     
        
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


