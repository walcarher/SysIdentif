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
def LatencyCPU(x, *b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = LinModel(x[1],b[3],b[4])
    k_model = QuadModel(x[2],b[5],b[6],b[7])
    N_model = LinModel(x[3],b[8],b[9])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model) 

def LatencyGPU(x, *b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = PolyModel(x[1],b[3],b[4],b[5],b[6])
    k_model = QuadModel(x[2],b[7],b[8],b[9])
    N_model = QuadModel(x[3],b[10],b[11],b[12])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)     

# FPGA Latency KPI estimation function from previous SI parameters
def LatencyFPGA(x, *b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = QuadModel(x[1],b[3],b[4],b[5])
    k_model = LinModel(x[2],b[6],b[7])
    N_model = QuadModel(x[3],b[8],b[9],b[10])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)     

# Resources
def ALM_FPGA(x, *b):
    HW_model = LinModel(x[0],b[0],b[1])
    C_model = PolyModel(x[1],b[2],b[3],b[4],b[5])
    k_model = PolyModel(x[2],b[6],b[7],b[8],b[9])
    N_model = QuadModel(x[3],b[10],b[11],b[12])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)
    
def ALUT_FPGA(x, *b):
    HW_model = LinModel(x[0],b[0],b[1])
    C_model = PolyModel(x[1],b[2],b[3],b[4],b[5])
    k_model = PolyModel(x[2],b[6],b[7],b[8],b[9])
    N_model = PolyModel(x[3],b[10],b[11],b[12],b[13])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)
    
def LAB_FPGA(x, *b):
    HW_model = LinModel(x[0],b[0],b[1])
    C_model = QuadModel(x[1],b[2],b[3],b[4])
    k_model = PolyModel(x[2],b[5],b[6],b[7],b[8])
    N_model = PolyModel(x[3],b[9],b[10],b[11],b[12])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)
    
def M20K_FPGA(x, *b):
    HW_model = LinModel(x[0],b[0],b[1])
    C_model = QuadModel(x[1],b[2],b[3],b[4])
    k_model = PolyModel(x[2],b[5],b[6],b[7],b[8])
    N_model = QuadModel(x[3],b[9],b[10],b[11])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)

def LatencyGPUFPGA_COMM(x, *b):
    return LinModel(x[0],b[0],b[1])

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
# CPU variables X_C
HW_C = cp.Variable(pos = True, name = "HW_C")
C_C = cp.Variable(pos = True, name = "C_C")
k_C = cp.Variable(pos = True, name = "k_C")
N_C = cp.Variable(pos = True, name = "N_C")
# GPU variables X_G
HW_G = cp.Variable(pos = True, name = "HW_G")
C_G = cp.Variable(pos = True, name = "C_G")
k_G = cp.Variable(pos = True, name = "k_G")
N_G = cp.Variable(pos = True, name = "N_G")
# FPGA variables X_F
HW_F = cp.Variable(pos = True, name = "HW_F")
C_F = cp.Variable(pos = True, name = "C_F")
k_F = cp.Variable(pos = True, name = "k_F")
N_F = cp.Variable(pos = True, name = "N_F")
# FPGA constant constrains (For C10GX: 10CX220YF780E5G)
ALM_MAX = cp.Constant(80330) # Max number of Arithmetic Logic Modules
#ALUT_MAX = cp.Constant(name = "ALUT_MAX") # Max number of Adaptive Look-Up Table - Overlaps with ALMs
LAB_MAX = cp.Constant(8033) # Max number of Memory Logic Array Block
M20K_MAX = cp.Constant(587) # Max number of Memory M20K blocks
# CNN model: AlexNet model
#num_layers = 5
#layer_names =       ["Conv0","Conv1","Conv2","Conv3","Conv4"]
#HW_layers = np.array([224,54 ,26 ,12 ,12 ])
#C_layers = np.array ([3  ,96 ,256,384,384])
#k_layers = np.array ([11 ,5  ,3  ,3  ,3  ])
#N_layers = np.array ([96 ,256,384,384,256])
# CNN model: VGG16
#num_layers = 13
#layer_names =       ["L0","L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11","L12"]
#HW_layers = np.array ([224 ,224 ,112 ,112 ,56 ,56 ,56 ,28 ,28 ,28 ,14 ,14 ,14])
#C_layers = np.array ([3, 64 ,64 ,128 ,128 ,256 ,256 ,256 ,512 ,512 ,512 ,512 ,512])
#k_layers = np.array ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
#N_layers = np.array ([64 ,64 ,128 ,128 ,256 ,256 ,256 ,512 ,512 ,512 ,512 ,512 ,512])
# CNN model: ResNet18
num_layers = 17
layer_names =       ["L0","L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11","L12","L13","L14","L15","L16"]
HW_layers = np.array ([224, 56 ,56 ,56 ,56 ,56 ,28 ,28 ,28 ,28 ,14 ,14 ,14 ,14 ,7 ,7 ,7])
C_layers = np.array ([3, 64 ,64 ,64 ,64 ,64 ,128 ,128 ,128 ,128 ,256 ,256 ,256 ,256 ,512 ,512 ,512])
k_layers = np.array ([7, 3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3])
N_layers = np.array ([64, 64 ,64 ,64 ,64 ,128 ,128 ,128 ,128 ,256 ,256 ,256 ,256 ,512 ,512 ,512 ,512])
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

C_C_list, C_G_list, C_F_list, eq_const_list, obj_results = [], [], [], [], []
CPU_lat, GPU_lat, FPGA_lat, COMM_lat = [], [], [], []
singleDev_results = []

for l in range(num_layers):
    # Tensor to be partitionned  (Example GConvv 224x224x3 with 32 filters of size 3x3)
    HW = cp.Constant(HW_layers[l])
    C = cp.Constant(C_layers[l])
    k = cp.Constant(k_layers[l])
    N = cp.Constant(N_layers[l])
    C_C_h = cp.Constant(C_layers[l]/3)
    C_F_h = cp.Constant(C_layers[l]/3)
    C_G_h = cp.Constant(C_layers[l]/3)
    # Constraints definition                                         
    constraints = [HW_C>=1,C_C>=0.1,k_C>=1,N_C>=1,
                   HW_G>=1,C_G>=0.1,k_G>=1,N_G>=1,
                   HW_F>=1,C_F>=0.1,k_F>=1,N_F>=1,
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
                   #ALM_FPGA([HW_F, C_F, k_F, N_F], *constantsALM) <= ALM_MAX,
                   #LAB_FPGA([HW_F, C_F, k_F, N_F], *constantsLAB) <= LAB_MAX,
                   #M20K_FPGA([HW_F, C_F, k_F, N_F], *constantsM20K) <= M20K_MAX,
                   # Relaxed constraints
                   (C_C+C_G+C_F)/C <= 1, # Relaxation from C_C + C_F + C_G == C
                   ]
    # Sweep over different W_k weight values
    w = 0
    steps = 0
    step_size = 10
    last_eq_value = 0.0
    while last_eq_value <= 0.99:
        # Forming penalization term
        steps = steps + 1
        w = w + step_size
        W = cp.Constant(w)
        exponent0 = C_C_h/(C_C_h+C_F_h+C_G_h)
        exponent1 = C_F_h/(C_C_h+C_F_h+C_G_h)
        exponent2 = C_G_h/(C_C_h+C_F_h+C_G_h)
        condensation0 = cp.power(C_C/C/(C_C_h/(C_C_h+C_F_h+C_G_h)),exponent0.value)
        condensation1 = cp.power(C_F/C/(C_F_h/(C_C_h+C_F_h+C_G_h)),exponent1.value)
        condensation2 = cp.power(C_G/C/(C_G_h/(C_C_h+C_F_h+C_G_h)),exponent2.value)
        penalization = 1 / (condensation0*condensation1*condensation2)
        # Heterogeneous objective function (Lateny in ms) (Sequential => addition) (Concurrent => max function)
        objective_fn = 1000*LatencyCPU([HW_C, C_C, k_C, N_C], *constantsCPU) + \
                       1000*LatencyGPU([HW_G, C_G, k_G, N_G], *constantsGPU) + \
                       LatencyFPGA([HW_F, C_F, k_F, N_F], *constantsFPGA) + \
                       LatencyGPUFPGA_COMM([HW_F*HW_F*C_F*8/1024], *constantsGPUFPGACOMM) + \
                       W*penalization
        # objective_fn = cp.maximum(1000*LatencyCPU([HW_C, C_C, k_C, N_C], *constantsCPU),
                                  # 1000*LatencyGPU([HW_G, C_G, k_G, N_G], *constantsGPU),
                                  # LatencyFPGA([HW_F, C_F, k_F, N_F], *constantsFPGA)+ \
                                  # LatencyGPUFPGA_COMM([HW_F*HW_F*C_F*8/1024], *constantsGPUFPGACOMM)) + \
                       # W*penalization
        # Minimize convex objective function                
        objective = cp.Minimize(objective_fn)
        prob = cp.Problem(objective, constraints)
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
        #last_eq_value = (np.rint(C_C.value)+np.rint(C_F.value)+np.rint(C_G.value))/C.value
        last_eq_value = (C_C.value+C_F.value+C_G.value)/C.value
    
    # Appending results
    C_C_list.append(C_C.value/C.value)
    C_F_list.append(C_F.value/C.value)
    C_G_list.append(C_G.value/C.value)
    eq_const_list.append(last_eq_value)
    obj_results.append((opt_val-W.value*penalization.value)/1000)
    singleDev_results.append(LatencyCPU([HW_layers[l], C_layers[l], k_layers[l], N_layers[l]], *constantsCPU).value)
    CPU_lat.append(LatencyCPU([HW_C, C_C, k_C, N_C], *constantsCPU).value)
    GPU_lat.append(LatencyGPU([HW_G, C_G, k_G, N_G], *constantsGPU).value)
    FPGA_lat.append(LatencyFPGA([HW_F, C_F, k_F, N_F], *constantsFPGA).value/1000)
    COMM_lat.append(LatencyGPUFPGA_COMM([HW_F*HW_F*C_F*8/1024], *constantsGPUFPGACOMM))
    # Printing Results
    print("Solution found: ", opt_val-W.value*penalization.value, " for layer: ", layer_names[l])
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

# Plot results
width = 0.5
fig, ax = plt.subplots()
rects0 = ax.bar(layer_names[0:num_layers], np.asarray(C_C_list), width, label='CPU ' r'($C_C$)', color = 'tomato', bottom = (np.asarray(C_F_list)+np.asarray(C_G_list)))
rects1 = ax.bar(layer_names[0:num_layers], np.asarray(C_F_list), width, label='FPGA ' r'($C_F$)', color = 'blue')
rects2 = ax.bar(layer_names[0:num_layers], np.asarray(C_G_list), width, label='GPU ' r'($C_G$)', color = 'green', bottom = np.asarray(C_F_list))
ax.set_ylabel('Normalized number of Channels ' r'$\frac{C_C+C_F+C_G}{C}$', color = 'black', fontweight = 'bold')
ax.set_xlabel('Layer name', color = 'black', fontweight = 'bold')
ax.grid()
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=3)
x = np.arange(num_layers)
fig2, ax2 = plt.subplots()
ax2.plot(x,np.asarray(CPU_lat),'-o',markersize=10, color='tomato',linewidth=2,label='CPU')
ax2.plot(x,np.asarray(GPU_lat),'-s',markersize=10, color='green',linewidth=2,label='GPU')
ax2.plot(x,np.asarray(FPGA_lat),'-^',markersize=10, color='blue',linewidth=3,label='FPGA')
ax2.plot(x,np.asarray(singleDev_results),'--',markersize=10, color='gray',linewidth=2,label='Single device')
ax2.plot(x,np.asarray(obj_results),'-D',markersize=10, color='purple',linewidth=2,label='Heterogeneous')
ax2.set_ylabel('Latency ' r'($LAT_{Het}$ in $ms$)', color = 'black', fontweight = 'bold')
ax2.set_xlabel('Layer name', color = 'black', fontweight = 'bold')
plt.xticks(np.arange(num_layers), layer_names[0:num_layers])
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=5)
ax3 = ax2.twinx()
ax3.set_ylabel('Speed-up factor', color='red', fontweight = 'bold')
ax3.plot(x, np.asarray(singleDev_results)/np.asarray(obj_results),'-*',markersize=12, linewidth=2, color='red')
ax3.tick_params(axis='y', labelcolor='red')
fig.tight_layout() 
plt.show()