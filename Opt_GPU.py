import sys
import cvxpy as cp
import numpy as np
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
    
# GPU Latency KPI estimation function from previous SI parameters
def LatencyGPU(x ,*b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = PolyModel(x[1],b[3],b[4],b[5],b[6])
    k_model = QuadModel(x[2],b[7],b[8],b[9])
    N_model = QuadModel(x[3],b[10],b[11],b[12])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)     

# FPGA Latency KPI estimation function from previous SI parameters
def LatencyFPGA(x ,*b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = QuadModel(x[1],b[3],b[4],b[5])
    k_model = LinModel(x[2],b[6],b[7])
    N_model = QuadModel(x[3],b[8],b[9],b[10])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)     


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

# Construct the problem.
# Variable tensors dimensions per device
# GPU variables
HW_G = cp.Variable(pos = True, name = "HW_G")
C_G = cp.Variable(pos = True, name = "C_G")
k_G = cp.Variable(pos = True, name = "k_G")
N_G = cp.Variable(pos = True, name = "N_G")
# FPGA variables
HW_F = cp.Variable(pos = True, name = "HW_F")
C_F = cp.Variable(pos = True, name = "C_F")
k_F = cp.Variable(pos = True, name = "k_F")
N_F = cp.Variable(pos = True, name = "N_F")
# FPGA constant constrains (For C10GX: 10CX220YF780E5G)
ALM_MAX = cp.Constant(80330) # Max number of Arithmetic Logic Modules
#ALUT_MAX = cp.Constant(name = "ALUT_MAX") # Max number of Adaptive Look-Up Table - Overlaps with ALMs
LAB_MAX = cp.Constant(8033) # Max number of Memory Logic Array Block
M20K_MAX = cp.Constant(587) # Max number of Memory M20K blocks
# Tensor to be partitionned  (Example 224x224x3 with 32 filters of size 3x3)
HW = cp.Constant(224)
C = cp.Constant(3)
k = cp.Constant(1)
N = cp.Constant(32)
# Device parameters/coefficients
constantsGPU = cp.Constant(parametersLATGPU)
constantsFPGA = cp.Constant(parametersLATFPGA)
# Print strong regresor model parameters on each device
print("GPU Parameters : ", constantsGPU)
print("FPGA Parameters : ", constantsFPGA)
# Heterogeneous objective function (Lateny in ms)
objective_fn = 1000*LatencyGPU([HW_G, C_G, k_G, N_G], *constantsGPU) + \
               LatencyFPGA([HW_F, C_F, k_F, N_F], *constantsFPGA)
# Constraints definition                                         
constraints = [HW_G>=1,C_G>=1,k_G>=1,N_G>=1,HW_F>=1,C_F>=1,k_F>=1,N_F>=1,
               HW_G == HW,
               HW_F == HW,
               k_G == k,
               k_F == k,
               N_G == N,
               N_F == N,
               C_G + C_F <= C # ERROR: this is not a monomial. GP is not possible
               ]
               #HW_G + HW_F == HW,
               #C_G + C_F == C,
               #k_G + k_F == k,
               #N_G + N_F == N]
#assert objective_fn.is_log_log_convex()
#assert all(constraint.is_dgp() for constraint in constraints)
objective = cp.Minimize(objective_fn)
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
if prob.is_dgp() == True:
    print("Using GP")
    result = prob.solve(gp=True)
elif prob.is_dcp() == True:
    print("Using CP")
    result = prob.solve()
else:
    print("Problem is non-convex")
    try:
        prob.solve()
    except cp.DCPError as e:
        print(e)
    sys.exit()

# The optimal value for x is stored in `x.value`.
print("Solution found: ", prob.value)
print("Solver used: ", prob.solver_stats.solver_name)
print("GPU feature values")
print("Value for", HW_G, "feature: ", HW_G.value)
print("Value for", C_G, "feature: ", C_G.value)
print("Value for", k_G, "feature: ", k_G.value)
print("Value for", N_G, "feature: ", N_G.value)
print("Value for", HW_F, "feature: ", HW_F.value)
print("Value for", C_F, "feature: ", C_F.value)
print("Value for", k_F, "feature: ", k_F.value)
print("Value for", N_F, "feature: ", N_F.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
#print(constraints[0].dual_value)