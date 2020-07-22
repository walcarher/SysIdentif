import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load previously generated dataset from DataGen.py
file = open('datasetPRBS.pkl', 'rb')
if not file:
    sys.exit("No .pkl file was found")
else:
    MMACs_list, LAT_list, POW_list= pickle.load(file)
if not MMACs_list or not LAT_list or not POW_list:
    sys.exit("Data loaded was empty from .pkl file")

L = len(LAT_list)
t = np.linspace(1, L, L, dtype = int ).tolist()

# ----------------- LATENCY SYSTEM IDENTIFICATION ----------------------------

# Data centering and normalization
u = np.array(MMACs_list) 
u = u - u.mean()
#u = (u - np.min(u)) / (np.max(u) -np.min(u))
y = np.array(LAT_list) 
y = y - y.mean()
#y = (y - np.min(y)) / (np.max(y) -np.min(y))

# Preprocessed data visualization
fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t, u, color='b')
ax2.plot(t, y, color='g')
plt.title('System Latency Response to BPRS', fontsize = 20)
ax1.set_xlabel('Sample', fontsize = 18)
ax1.set_ylabel('Centered #MACs (M)', fontsize = 18, color='b')
ax2.set_ylabel('Centered Latency (ms)', fontsize = 18, color='g')
plt.grid()

# Offline system SISO linear identification (none or small noise)
# 0-Order (no delay)
V = []
M = []
theta = []
V = y[0:L]
M = np.column_stack((-np.ones_like(y[0:L]), u[0:L]))
theta = np.matmul(np.linalg.pinv(M),V)
print(theta)
a0 = theta[0]
b0 = theta[1]
y_h0 = np.zeros_like(y)
i = 0
for i in range(L): 
	y_h0[i] = -a0 + b0*u[i]

plt.figure()
plt.plot(t, y, label='System output', color='g')
plt.plot(t, y_h0, label='Model output', color='r')
plt.title('Latency 0-Order Model', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Latency (ms)', fontsize = 18)
plt.legend()
plt.grid()

# 1st Order
V = []
M = []
theta = []
V = y[1:L]
M = np.column_stack((-y[0:L-1], u[0:L-1]))
theta = np.matmul(np.linalg.pinv(M),V)
print(theta)
a1 = theta[0]
b1 = theta[1]
y_h1 = np.zeros_like(y)
y_h1[0] = y[0]
i = 1
for i in range(L): 
	y_h1[i] = -a1*y_h1[i-1] + b1*u[i-1]

plt.figure()
plt.plot(t, y, label='System output', color='g')
plt.plot(t, y_h1, label='Model output', color='r')
plt.title('Latency 1st-Order Model', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Latency (ms)', fontsize = 18)
plt.legend()
plt.grid()

# 2nd Order
V = []
M = []
theta = []
V = y[2:L]
M = np.column_stack((-y[1:L-1], -y[0:L-2], u[1:L-1], u[0:L-2]))
theta = np.matmul(np.linalg.pinv(M),V)
print(theta)
a21 = theta[0]
a22 = theta[1]
b21 = theta[2]
b22 = theta[3]
y_h2 = np.zeros_like(y)
y_h2[0:2] = (y[0], y[1])
i = 2
for i in range(L): 
	y_h2[i] = -a21*y_h2[i-1] - a22*y_h2[i-2] + b21*u[i-1] + b22*u[i-2]

plt.figure()
plt.plot(t, y, label='System output', color='g')
plt.plot(t, y_h2, label='Model output', color='r')
plt.title('Latency 2nd-Order Model', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Latency (ms)', fontsize = 18)
plt.legend()
plt.grid()

# 3nd Order
V = []
M = []
theta = []
V = y[3:L]
M = np.column_stack((-y[2:L-1], -y[1:L-2], -y[0:L-3], u[2:L-1], u[1:L-2], u[0:L-3]))
theta = np.matmul(np.linalg.pinv(M),V)
print(theta)
a31 = theta[0]
a32 = theta[1]
a33 = theta[2]
b31 = theta[3]
b32 = theta[4]
b33 = theta[5]
y_h3 = np.zeros_like(y)
y_h3[0:3] = (y[0], y[1], y[2])
i = 3
for i in range(L): 
	y_h3[i] = -a31*y_h3[i-1] - a32*y_h3[i-2] - a33*y_h3[i-3] + b31*u[i-1] + b32*u[i-2] + b33*u[i-3]

plt.figure()
plt.plot(t, y, label='System output', color='g')
plt.plot(t, y_h3, label='Model output', color='r')
plt.title('Latency 3rd Order Model', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Latency (ms)', fontsize = 18)
plt.legend()
plt.grid()

# n-th Order

# ----------------- ENERGY SYSTEM IDENTIFICATION ----------------------------

# Element-wise product for energy measu 
E_list = np.multiply(LAT_list, POW_list)

#u = (u - np.min(u)) / (np.max(u) -np.min(u))
y = np.array(E_list) 
y = y - y.mean()
#y = (y - np.min(y)) / (np.max(y) -np.min(y))

# Preprocessed data visualization
fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t, u, color='b')
ax2.plot(t, y, color='g')
plt.title('System Energy Response to BPRS', fontsize = 20)
ax1.set_xlabel('Sample', fontsize = 18)
ax1.set_ylabel('Centered #MACs (M)', fontsize = 18, color='b')
ax2.set_ylabel('Centered Energy (mJ)', fontsize = 18, color='g')
plt.grid()

# Offline system SISO linear identification (none or small noise)
# 0-Order (no delay)
V = []
M = []
theta = []
V = y[0:L]
M = np.column_stack((-np.ones_like(y[0:L]), u[0:L]))
theta = np.matmul(np.linalg.pinv(M),V)
print(theta)
a0 = theta[0]
b0 = theta[1]
y_h0 = np.zeros_like(y)
i = 0
for i in range(L): 
	y_h0[i] = -a0 + b0*u[i]

plt.figure()
plt.plot(t, y, label='System output', color='g')
plt.plot(t, y_h0, label='Model output', color='r')
plt.title('Energy 0-Order Model', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Energy (mJ)', fontsize = 18)
plt.legend()
plt.grid()

# 1st Order
V = []
M = []
theta = []
V = y[1:L]
M = np.column_stack((-y[0:L-1], u[0:L-1]))
theta = np.matmul(np.linalg.pinv(M),V)
print(theta)
a1 = theta[0]
b1 = theta[1]
y_h1 = np.zeros_like(y)
y_h1[0] = y[0]
i = 1
for i in range(L): 
	y_h1[i] = -a1*y_h1[i-1] + b1*u[i-1]

plt.figure()
plt.plot(t, y, label='System output', color='g')
plt.plot(t, y_h1, label='Model output', color='r')
plt.title('Energy 1st-Order Model', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Energy (mJ)', fontsize = 18)
plt.legend()
plt.grid()

# 2nd Order
V = []
M = []
theta = []
V = y[2:L]
M = np.column_stack((-y[1:L-1], -y[0:L-2], u[1:L-1], u[0:L-2]))
theta = np.matmul(np.linalg.pinv(M),V)
print(theta)
a21 = theta[0]
a22 = theta[1]
b21 = theta[2]
b22 = theta[3]
y_h2 = np.zeros_like(y)
y_h2[0:2] = (y[0], y[1])
i = 2
for i in range(L): 
	y_h2[i] = -a21*y_h2[i-1] - a22*y_h2[i-2] + b21*u[i-1] + b22*u[i-2]

plt.figure()
plt.plot(t, y, label='System output', color='g')
plt.plot(t, y_h2, label='Model output', color='r')
plt.title('Energy 2nd-Order Model', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Energy (mJ)', fontsize = 18)
plt.legend()
plt.grid()

# 3nd Order
V = []
M = []
theta = []
V = y[3:L]
M = np.column_stack((-y[2:L-1], -y[1:L-2], -y[0:L-3], u[2:L-1], u[1:L-2], u[0:L-3]))
theta = np.matmul(np.linalg.pinv(M),V)
print(theta)
a31 = theta[0]
a32 = theta[1]
a33 = theta[2]
b31 = theta[3]
b32 = theta[4]
b33 = theta[5]
y_h3 = np.zeros_like(y)
y_h3[0:3] = (y[0], y[1], y[2])
i = 3
for i in range(L): 
	y_h3[i] = -a31*y_h3[i-1] - a32*y_h3[i-2] - a33*y_h3[i-3] + b31*u[i-1] + b32*u[i-2] + b33*u[i-3]

plt.figure()
plt.plot(t, y, label='System output', color='g')
plt.plot(t, y_h3, label='Model output', color='r')
plt.title('Energy 3rd Order Model', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Energy (mJ)', fontsize = 18)
plt.legend()
plt.grid()

plt.show()
