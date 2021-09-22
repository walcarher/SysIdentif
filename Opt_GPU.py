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
    
# GPU Energy KPI estimation function from previous SI parameters
def LatencyGPU(x ,*b):
    HW_model = QuadModel(x[0],b[0],b[1],b[2])
    C_model = PolyModel(x[1],b[3],b[4],b[5],b[6])
    k_model = QuadModel(x[2],b[7],b[8],b[9])
    N_model = QuadModel(x[3],b[10],b[11],b[12])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)     

file = open('parametersLATGPU.pkl', 'rb')
if not file:
    sys.exit("No parametersLATGPU.pkl file was found")
else:
    parametersLATGPU = pickle.load(file)

# Construct the problem.
HW = cp.Variable(1, pos = True, name = "HW")
C = cp.Variable(1, pos = True, name = "C")
k = cp.Variable(1, pos = True, name = "k")
N = cp.Variable(1, pos = True, name = "N")
constantsGPU = cp.Constant(parametersLATGPU)
# Print strong regressor model parameters on each device
print("GPU Parameters : ", constantsGPU)
objective_fn = LatencyGPU([HW, C, k, N], *constantsGPU)
#objective_fn = LatencyGPUtest([HW, C, k, N], constant0, constant1, constant2, constant3,
#                                             constant4, constant5, constant6, constant7)
# Constraints definition                                         
constraints = [HW >= 1, C >= 1, k >= 1, N >= 1, 
               HW <= 50, C <= 100, k <= 11, N <= 50]
#assert objective_fn.is_log_log_convex()
#assert all(constraint.is_dgp() for constraint in constraints)
objective = cp.Minimize(objective_fn)
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
if objective.is_dgp() == True:
    print("Using GP")
    result = prob.solve(gp=True)
elif objective.is_dcp() == True:
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
print("Value for", HW, "feature: ", HW.value)
print("Value for", C, "feature: ", C.value)
print("Value for", k, "feature: ", k.value)
print("Value for", N, "feature: ", N.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
#print(constraints[0].dual_value)