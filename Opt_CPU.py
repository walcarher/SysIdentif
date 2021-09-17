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
@ParameterNumber(1)
def LinModel(x, a0):
    return a0*x

# Quadratic model
@Name('Quadratic')
@ParameterNumber(2)
def QuadModel(x, a1, a0):
    return a1*np.power(x, 2) + a0*x

# Logarithmic model
@Name('Logarithmic')  
@ParameterNumber(1)
def LogModel(x, a0):
    return a0*np.log(x)
    
# Exponential model   
@Name('Exponential') 
@ParameterNumber(2)
def ExpModel(x, a1, a0):
    return a1*np.power(a0, x)
    
# Reciprocal model   
@Name('Reciprocal') 
@ParameterNumber(1)
def ReciModel(x, a0):
    return a0*(1/x)
    
# Polynomial model   
@Name('Polynomial') 
@ParameterNumber(3)
def PolyModel(x, a2, a1, a0):
    return a2*np.power(x, 3) + a1*np.power(x, 2) + a0*x
    
# GPU Energy KPI estimation function from previous SI parameters
def LatencyGPU(x ,*b):
    HW_model = QuadModel(x[0],b[0],b[1])
    C_model = PolyModel(x[1],b[2],b[3],b[4])
    k_model = QuadModel(x[2],b[5],b[6])
    N_model = LinModel(x[3],b[7])
    return cp.multiply(cp.multiply(cp.multiply(HW_model, C_model), k_model), N_model)

file = open('parametersLATGPU.pkl', 'rb')
if not file:
    sys.exit("No parametersECPU.pkl file was found")
else:
    parametersLATGPU = pickle.load(file)

# Problem data.
# m = 30
# n = 20
# np.random.seed(1)
# A = np.random.randn(m, n)
# b = np.random.randn(m)

# Construct the problem.
HW = cp.Variable(1)
C = cp.Variable(1)
k = cp.Variable(1)
N = cp.Variable(1)
#x = cp.Variable(n)
#objective = cp.Minimize(cp.sum_squares(A@x - b))
print(parametersLATGPU)
objective = cp.Minimize(LatencyGPU([HW, C, k, N], *parametersLATGPU))
#constraints = [0 <= x, x <= 1]
constraints = [HW >= 1, C >= 1, k >= 1, N >= 1]
prob = cp.Problem(objective, constraints)
print(objective.is_dcp())
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
#print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
#print(constraints[0].dual_value)