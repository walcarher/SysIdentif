import sys
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
def LinModel(x, a1, a0,):
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
    return a1*cp.log(x) + a0
    
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

# Flydels 
# NOTE: Not an elegant solution... selected models are added manually from
# NonLinSysIdent(FPGA/GPU)_Posynomial.py file... any more advanced python lambda programmer may have a better solution
# GPU Latency KPI estimation function from previous SI parameters
def LatencyCPU(HW,C,k,N,*b):
    HW_model = QuadModel(HW,b[0],b[1],b[2])
    C_model = LinModel(C,b[3],b[4])
    k_model = QuadModel(k,b[5],b[6],b[7])
    N_model = LinModel(N,b[8],b[9])
    return cp.sum(cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model))

def LatencyGPU(HW,C,k,N,*b):
    HW_model = QuadModel(HW,b[0],b[1],b[2])
    C_model = PolyModel(C,b[3],b[4],b[5],b[6])
    k_model = QuadModel(k,b[7],b[8],b[9])
    N_model = QuadModel(N,b[10],b[11],b[12])
    return cp.sum(cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model))    

# FPGA Latency KPI estimation function from previous SI parameters
def LatencyFPGA(HW,C,k,N,*b):
    HW_model = QuadModel(HW,b[0],b[1],b[2])
    C_model = QuadModel(C,b[3],b[4],b[5])
    k_model = LinModel(k,b[6],b[7])
    N_model = QuadModel(N,b[8],b[9],b[10])
    return cp.sum(cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model))   

# Resources
def ALM_FPGA(HW,C,k,N,*b):
    HW_model = LinModel(HW,b[0],b[1])
    C_model = PolyModel(C,b[2],b[3],b[4],b[5])
    k_model = PolyModel(k,b[6],b[7],b[8],b[9])
    N_model = QuadModel(N,b[10],b[11],b[12])
    return cp.sum(cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model))
    
def ALUT_FPGA(HW,C,k,N,*b):
    HW_model = LinModel(HW,b[0],b[1])
    C_model = PolyModel(C,b[2],b[3],b[4],b[5])
    k_model = PolyModel(k,b[6],b[7],b[8],b[9])
    N_model = PolyModel(N,b[10],b[11],b[12],b[13])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)
    
def LAB_FPGA(HW,C,k,N,*b):
    HW_model = LinModel(HW,b[0],b[1])
    C_model = QuadModel(C,b[2],b[3],b[4])
    k_model = PolyModel(k,b[5],b[6],b[7],b[8])
    N_model = PolyModel(N,b[9],b[10],b[11],b[12])
    return cp.sum(cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model))
    
def M20K_FPGA(HW,C,k,N,*b):
    HW_model = LinModel(HW,b[0],b[1])
    C_model = QuadModel(C,b[2],b[3],b[4])
    k_model = PolyModel(k,b[5],b[6],b[7],b[8])
    N_model = QuadModel(N,b[9],b[10],b[11])
    return cp.sum(cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model))

def LatencyGPUFPGA_COMM(HW, C, *b):
    return cp.sum(LinModel(8/1024*cp.multiply(cp.multiply(HW,HW_F),C),b[0],b[1]))

file = open('parametersLATCPU.pkl', 'rb')
if not file:
    sys.exit("No parametersLATCPU.pkl file was found")
else:
    parametersLATCPU = pickle.load(file)

file = open('parametersLATGPU.pkl', 'rb')
if not file:
    sys.exit("No parametersLATGPU.pkl file was found")
else:
    parametersLATGPU = pickle.load(file)
    
file = open('parametersLATFPGA.pkl', 'rb')
if not file:
    sys.exit("No parametersLATFPGA.pkl file was found")
else:
    parametersLATFPGA = pickle.load(file)
    
file = open('parametersALMFPGA.pkl', 'rb')
if not file:
    sys.exit("No parametersALMFPGA.pkl file was found")
else:
    parametersALMFPGA = pickle.load(file)
    
file = open('parametersALUTFPGA.pkl', 'rb')
if not file:
    sys.exit("No parametersALUTFPGA.pkl file was found")
else:
    parametersALUTFPGA = pickle.load(file)
    
file = open('parametersLABFPGA.pkl', 'rb')
if not file:
    sys.exit("No parametersLABFPGA.pkl file was found")
else:
    parametersLABFPGA = pickle.load(file)
    
file = open('parametersM20KFPGA.pkl', 'rb')
if not file:
    sys.exit("No parametersM20KFPGA.pkl file was found")
else:
    parametersM20KFPGA = pickle.load(file)

file = open('parametersLATCOMM.pkl', 'rb')
if not file:
    sys.exit("No parametersLATCOMM.pkl file was found")
else:
    parametersLATCOMM = pickle.load(file)

# Define the problem.
# Variable tensors dimensions per device
# Number of CNN layers
num_layers = 5
# CPU variables X_C
HW_C = cp.Variable(pos = True, name = "HW_C", shape = num_layers)
C_C = cp.Variable(pos = True, name = "C_C", shape = num_layers)
k_C = cp.Variable(pos = True, name = "k_C", shape = num_layers)
N_C = cp.Variable(pos = True, name = "N_C", shape = num_layers)
# GPU variables X_G
HW_G = cp.Variable(pos = True, name = "HW_G", shape = num_layers)
C_G = cp.Variable(pos = True, name = "C_G", shape = num_layers)
k_G = cp.Variable(pos = True, name = "k_G", shape = num_layers)
N_G = cp.Variable(pos = True, name = "N_G", shape = num_layers)
# FPGA variables X_F
HW_F = cp.Variable(pos = True, name = "HW_F", shape = num_layers)
C_F = cp.Variable(pos = True, name = "C_F", shape = num_layers)
k_F = cp.Variable(pos = True, name = "k_F", shape = num_layers)
N_F = cp.Variable(pos = True, name = "N_F", shape = num_layers)
# Initializing variable values
init = np.ones(num_layers)
HW_C.value = HW_C.project(init)
C_C.value = C_C.project(np.array([3,96,256,384,384])/3)
k_C.value = k_C.project(init)
N_C.value = N_C.project(init)
HW_G.value = HW_G.project(init)
C_G.value = C_G.project(np.array([3,96,256,384,384])/3)
k_G.value = k_G.project(init)
N_G.value = N_G.project(init)
HW_F.value = HW_F.project(init)
C_F.value = C_F.project(np.array([3,96,256,384,384])/3)
k_F.value = k_F.project(init)
N_F.value = N_F.project(init)
# FPGA constant constrains (For C10GX: 10CX220YF780E5G)
ALM_MAX = cp.Constant(1000*80330) # Max number of Arithmetic Logic Modules
#ALM_MAX = cp.Constant(80330) # Max number of Arithmetic Logic Modules
#ALUT_MAX = cp.Constant(name = "ALUT_MAX") # Max number of Adaptive Look-Up Table - Overlaps with ALMs
LAB_MAX = cp.Constant(1000*8033) # Max number of Memory Logic Array Block
#LAB_MAX = cp.Constant(8033) # Max number of Memory Logic Array Block
M20K_MAX = cp.Constant(1000*587) # Max number of Memory M20K blocks
#M20K_MAX = cp.Constant(587) # Max number of Memory M20K blocks
# Tensor to be partitionned  (Example: AlexNet 224x224x3)
HW = cp.Constant([224,54,26,12,12])
C = cp.Constant([3,96,256,384,384])
k = cp.Constant([11,5,3,3,3])
N = cp.Constant([96,256,384,384,256])
C_C_h = cp.Constant([1,32,86,128,128])
C_F_h = cp.Constant([1,32,84,128,128])
C_G_h = cp.Constant([1,32,84,128,128])
# Device parameters/coefficients
constantsCPU = cp.Constant(parametersLATCPU)
constantsGPU = cp.Constant(parametersLATGPU)
constantsFPGA = cp.Constant(parametersLATFPGA)
# Device resources 
constantsALM = cp.Constant(parametersALMFPGA)
constantsALUT = cp.Constant(parametersALUTFPGA)
constantsLAB = cp.Constant(parametersLABFPGA)
constantsM20K = cp.Constant(parametersM20KFPGA)
# GPU-FPGA Communication parameters/coefficients
constantsGPUFPGACOMM = cp.Constant(parametersLATCOMM)
# Print strong regresor model parameters on each device
print("CPU Latency Parameters : ", constantsCPU)
print("GPU Latency Parameters : ", constantsGPU)
print("FPGA Latency Parameters : ", constantsFPGA)
print("GPU-FPGA Latency Parameters : ", constantsGPUFPGACOMM)
# Constraints definition                                         
constraints = [HW_C>=1,C_C>=1,k_C>=1,N_C>=1,
               HW_G>=1,C_G>=1,k_G>=1,N_G>=1,
               HW_F>=1,C_F>=1,k_F>=1,N_F>=1,
               HW_C == HW,
               HW_G == HW,
               HW_F == HW,
               k_C == k,
               k_G == k,
               k_F == k,
               N_C == N,
               N_G == N,
               N_F == N,
               # Resources constraints
               #ALM_FPGA(HW_F, C_F, k_F, N_F, *constantsALM) <= ALM_MAX,
               #LAB_FPGA(HW_F, C_F, k_F, N_F, *constantsLAB) <= LAB_MAX,
               #M20K_FPGA(HW_F, C_F, k_F, N_F, *constantsM20K) <= M20K_MAX,
               # Relaxed constraints
               #(C_C+C_G+C_F) <= C, # Relaxation from C_C + C_F + C_G == C
               ]
 
# Sweep over different W_k weight values
w = np.zeros(num_layers)
steps = 0
step_size = 10
C_C_list, C_G_list, C_F_list, eq_const_list, obj_results = [], [], [], [], []
last_eq_values = w
while np.any(last_eq_values <= 0.99):
    # Forming penalization term
    #print(C_C.value,C_F.value,C_G.value)
    steps = steps + 1
    for i,last_eq_value in np.ndenumerate(last_eq_values):
        if last_eq_value <= 0.99:
            w[i] = w[i] + step_size
    W = cp.Constant(w)
    exponent0 = C_C_h/(C_C_h+C_F_h+C_G_h)
    exponent1 = C_F_h/(C_C_h+C_F_h+C_G_h)
    exponent2 = C_G_h/(C_C_h+C_F_h+C_G_h)
    condensation0 = (C_C/C/(C_C_h/(C_C_h+C_F_h+C_G_h))).value**exponent0.value
    condensation1 = (C_F/C/(C_F_h/(C_C_h+C_F_h+C_G_h))).value**exponent1.value
    condensation2 = (C_G/C/(C_G_h/(C_C_h+C_F_h+C_G_h))).value**exponent2.value
    penalization = 1 / cp.multiply(cp.multiply(condensation0,condensation1),condensation2)
    # Heterogeneous objective function (Lateny in ms) (Sequential => addition) (Concurrent => max function)
    objective_fn = 1000*LatencyCPU(HW_C, C_C, k_C, N_C, *constantsCPU) + \
                   1000*LatencyGPU(HW_G, C_G, k_G, N_G, *constantsGPU) + \
                   LatencyFPGA(HW_F, C_F, k_F, N_F, *constantsFPGA) + \
                   LatencyGPUFPGA_COMM(HW_F, C_F, *constantsGPUFPGACOMM) + \
                   cp.sum(cp.multiply(W,penalization))
    # objective_fn = cp.maximum(1000*LatencyCPU([HW_C, C_C, k_C, N_C], *constantsCPU),
                              # 1000*LatencyGPU([HW_G, C_G, k_G, N_G], *constantsGPU),
                              # LatencyFPGA([HW_F, C_F, k_F, N_F], *constantsFPGA)) + \
                   # W*penalization
    # Minimize convex objective function                
    objective = cp.Minimize(objective_fn)
    prob = cp.Problem(objective, constraints)
    #print(prob)
    # The optimal objective value is returned by `prob.solve()`.
    if prob.is_dgp() == True:
        #print("Using GP")
        result = prob.solve(gp = True)
    elif prob.is_dcp() == True:
        #print("Using CP")
        result = prob.solve()
    else:
        #print("Problem is Non-Convex")
        try:
            prob.solve()
        except cp.DCPError as e:
            print(e)
        sys.exit()   
    # The optimal value for x is stored in `x.value`.  
    # Store optimal value
    opt_val = prob.value 
    #Appending results
    #C_C_list.append(np.rint(C_C.value))
    #C_F_list.append(np.rint(C_F.value))
    #C_G_list.append(np.rint(C_G.value))
    last_eq_values = (C_C.value+C_F.value+C_G.value)/C.value
    print(C_C.value,C_F.value,C_G.value)
    #eq_const_list.append(last_eq_values)
    #obj_results.append(opt_val-W.value*penalization.value)

print("Solution found: ", opt_val)
print("Solver used: ", prob.solver_stats.solver_name)
print("CPU feature values")
print("Value for", HW_C, "feature: ", np.rint(HW_C.value))
print("Value for", C_C, "feature: ", np.rint(C_C.value))
print("Value for", k_C, "feature: ", np.rint(k_C.value))
print("Value for", N_C, "feature: ", np.rint(N_C.value))
print("GPU feature values")
print("Value for", HW_G, "feature: ", np.rint(HW_G.value))
print("Value for", C_G, "feature: ", np.rint(C_G.value))
print("Value for", k_G, "feature: ", np.rint(k_G.value))
print("Value for", N_G, "feature: ", np.rint(N_G.value))
print("FPGA feature values")
print("Value for", HW_F, "feature: ", np.rint(HW_F.value))
print("Value for", C_F, "feature: ", np.rint(C_F.value))
print("Value for", k_F, "feature: ", np.rint(k_F.value))
print("Value for", N_F, "feature: ", np.rint(N_F.value))
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
#print(constraints[0].dual_value)

# Plot results
# width = 1.75*step_size/steps
# x = np.linspace(1, steps, len(eq_const_list))
# y = np.asarray(eq_const_list)
# fig, ax = plt.subplots()
# rects0 = ax.bar(x, np.asarray(C_C_list), width, label='CPU ' r'($C_C$)', color = 'tomato', bottom = (np.asarray(C_F_list)+np.asarray(C_G_list)))
# rects1 = ax.bar(x, np.asarray(C_F_list), width, label='FPGA ' r'($C_F$)', color = 'blue')
# rects2 = ax.bar(x, np.asarray(C_G_list), width, label='GPU ' r'($C_G$)', color = 'green', bottom = np.asarray(C_F_list))
# ax.set_ylabel('Number of Channels ' r'($C_C+C_F+C_G$)', color = 'black', fontweight = 'bold')
# ax.set_xlabel('Penalization weight ' r'($\alpha$)', color = 'black', fontweight = 'bold')
# ax.grid()
# ax.legend()
# ax2 = ax.twinx()
# color = 'tab:red'
# ax2.plot(x,y,'-',color='red',linewidth=2.5)
# ax2.set_ylabel('Equality constrain value', color = color, fontweight = 'bold')
# ax2.tick_params(axis='y', labelcolor=color)
# fig, ax3 = plt.subplots()
# ax3.plot(x,np.asarray(obj_results),'-',color='purple',linewidth=2.5)
# ax3.set_ylabel('Latency ' r'($LAT_{Het}$ in ms)', color = 'black', fontweight = 'bold')
# ax3.set_xlabel('Penalization weight ' r'($\alpha$)', color = 'black', fontweight = 'bold')
# ax3.grid()
# plt.show()