import sys
import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
    
# This files are to be found in the L4T version of TX2, this may vary in the future
# It reads the latest GPU Power in mW from the INA3221 in the TegraTX2
gpuLoadFile = '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power0_input'

# Load parameters to test
file = open('parametersEGPU.pkl', 'rb')
if not file:
    sys.exit("No parametersEGPU.pkl file was found")
else:
    parameters = pickle.load(file)
    
# Enenergy KPI estimation function from previous SI parameters
def EnergyEst(HW, C, k, N):
    return parameters[0]*(HW**2)+parameters[1]*(HW)+parameters[2]+parameters[3]*(C**3)+parameters[4]*(C**2)+parameters[5]*(C) + \
    parameters[6]+parameters[7]*(k**3)+parameters[8]*(k**2)+parameters[9]*(k)+parameters[10]+parameters[11]*(N**2)+parameters[12]*(N)+parameters[13]+parameters[14] 
    
input_tensor = 32 # input image/tensor size i.e. 224x224 for ImageNet
input_channel = 100
#start_num_convs = 100 # starting number of filters or depth of the output tensor
#max_num_convs = 3000 # max number of filters or depth of the output tensor
num_conv_list = [200, 300, 400, 500, 600]
#step_size_convs = 100 # step size from start to maximum number of iterations
n_iter = 5000  # Number of iterations on a single convolution run
	       # the average of results is reported in output file
time_delay = 0.2 # Pause between running tests

# Writing results to this file
#csv_file = open('GPU_time.csv', "wb")

# Random input tensor or image with 1 batch, 1 channel and size 
# input_tensorxinput_tensor
input = torch.rand(1,input_channel, input_tensor, input_tensor).cuda()

#def iter_range(start, end, step):
#	while start <= end:
#		yield start
#		start += step

power = []
latency = []
energyEst = []

#for num_convs in iter_range(start_num_convs, max_num_convs, step_size_convs):
for num_convs in num_conv_list:

    print("Number of convolutions: %d" % num_convs)

    class Conv1x1_Net(nn.Module):

        def __init__(self):
            super(Conv1x1_Net,self).__init__()
            #1 batch size, n conv output, kernel size 1x1, stride 1-1
            self.conv1 = nn.Conv2d(input_channel, num_convs, 1).cuda()

        def forward(self, x):
            # Maxpooling 2x2 and ReLu activation function
            #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
            x = self.conv1(x).cuda()
            return x

    class Conv3x3_Net(nn.Module):

        def __init__(self):
            super(Conv3x3_Net,self).__init__()
            #1 batch size, n conv output, kernel size 3x3, stride 1-1
            self.conv1 = nn.Conv2d(input_channel, num_convs, 3).cuda()

        def forward(self, x):
            # Maxpooling 2x2 and ReLu activation function
            #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
            x = self.conv1(x).cuda()			
            return x

    class Conv5x5_Net(nn.Module):

        def __init__(self):
            super(Conv5x5_Net,self).__init__()
            #1 batch size, n conv output, kernel size 5x5, stride 1-1
            self.conv1 = nn.Conv2d(input_channel, num_convs, 5).cuda()

        def forward(self, x):
            # Maxpooling 2x2 and ReLu activation function
            #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
            x = self.conv1(x).cuda()			
            return x

    class Conv7x7_Net(nn.Module):

        def __init__(self):
            super(Conv7x7_Net,self).__init__()
            #1 batch size, n conv output, kernel size 7x7, stride 1-1
            self.conv1 = nn.Conv2d(input_channel, num_convs, 7).cuda()

        def forward(self, x):
            # Maxpooling 2x2 and ReLu activation function
            #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
            x = self.conv1(x).cuda()			
            return x

    class Conv11x11_Net(nn.Module):

        def __init__(self):
            super(Conv11x11_Net,self).__init__()
            #1 batch size, n conv output, kernel size 11x11, stride 1-1
            self.conv1 = nn.Conv2d(input_channel, num_convs, 11).cuda()

        def forward(self, x):
            # Maxpooling 2x2 and ReLu activation function
            #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
            x = self.conv1(x).cuda()			
            return x

    # Convolution layer model
    conv1x1_net = Conv1x1_Net()
    print(conv1x1_net)
    time.sleep(time_delay)

    print("Now running ...")
    i = 0
    #csv_file.write(str(num_convs)+',')
    #csv_file.write('Conv1x1'+',')
    #csv_file.write(str(time.time())+',')
    while(i < n_iter):
        start = time.time()
        input = torch.rand(1,input_channel, input_tensor, input_tensor).cuda()
        out = conv1x1_net(input).cuda()
        with open(gpuLoadFile, 'r') as gpuFile:
            power.append(float(gpuFile.read()))
        torch.cuda.synchronize()
        latency.append(time.time() - start)
        energyEst.append(EnergyEst(input_tensor,input_channel,1,num_convs))
        i += 1
    #csv_file.write(str(time.time())+',')
    #csv_file.write('\n')

    # Convolution layer model
    conv3x3_net = Conv3x3_Net()
    print(conv3x3_net)

    time.sleep(time_delay)
    print("Now running ...")
    i = 0
    #csv_file.write(str(num_convs)+',')
    #csv_file.write('Conv3x3'+',')
    #csv_file.write(str(time.time())+',')
    while(i < n_iter):
        start = time.time()
        input = torch.rand(1,input_channel, input_tensor, input_tensor).cuda()
        out = conv3x3_net(input).cuda()
        with open(gpuLoadFile, 'r') as gpuFile:
            power.append(float(gpuFile.read()))
        torch.cuda.synchronize()
        latency.append(time.time() - start)
        energyEst.append(EnergyEst(input_tensor,input_channel,3,num_convs))
        i += 1
    #csv_file.write(str(time.time())+',')
    #csv_file.write('\n')

    # Convolution layer model
    conv5x5_net = Conv5x5_Net()
    print(conv5x5_net)

    time.sleep(time_delay)
    print("Now running ...")
    i = 0
    #csv_file.write(str(num_convs)+',')
    #csv_file.write('Conv5x5'+',')
    #csv_file.write(str(time.time())+',')
    while(i < n_iter):
        start = time.time()
        input = torch.rand(1,input_channel, input_tensor, input_tensor).cuda()
        out = conv5x5_net(input).cuda()
        with open(gpuLoadFile, 'r') as gpuFile:
            power.append(float(gpuFile.read()))
        torch.cuda.synchronize()
        latency.append(time.time() - start)
        energyEst.append(EnergyEst(input_tensor,input_channel,5,num_convs))
        i += 1
    #csv_file.write(str(time.time())+',')
    #csv_file.write('\n')

    # Convolution layer model
    conv7x7_net = Conv7x7_Net()
    print(conv7x7_net)

    time.sleep(time_delay)
    print("Now running ...")
    i = 0
    #csv_file.write(str(num_convs)+',')
    #csv_file.write('Conv7x7'+',')
    #csv_file.write(str(time.time())+',')
    while(i < n_iter):
        start = time.time()
        input = torch.rand(1,input_channel, input_tensor, input_tensor).cuda()
        out = conv7x7_net(input).cuda()
        with open(gpuLoadFile, 'r') as gpuFile:
            power.append(float(gpuFile.read()))
        torch.cuda.synchronize()
        latency.append(time.time() - start)
        energyEst.append(EnergyEst(input_tensor,input_channel,7,num_convs))
        i += 1
    #csv_file.write(str(time.time())+',')
    #csv_file.write('\n')

    # Convolution layer model
    conv11x11_net = Conv11x11_Net()
    print(conv11x11_net)

    time.sleep(time_delay)
    print("Now running ...")
    i = 0
    #csv_file.write(str(num_convs)+',')
    #csv_file.write('Conv11x11'+',')
    #csv_file.write(str(time.time())+',')
    while(i < n_iter):
        start = time.time()
        input = torch.rand(1,input_channel, input_tensor, input_tensor).cuda()
        out = conv11x11_net(input).cuda()
        with open(gpuLoadFile, 'r') as gpuFile:
            power.append(float(gpuFile.read()))
        torch.cuda.synchronize()
        latency.append(time.time() - start)
        energyEst.append(EnergyEst(input_tensor,input_channel,11,num_convs))
        i += 1
    #csv_file.write(str(time.time())+',')
    #csv_file.write('\n')

print("Test Done")

#HW = 64
#C = 3
#k_s = np.concatenate((np.full((1,100), 1), np.full((1,100), 3), np.full((1,100), 5), np.full((1,100), 7), np.full((1,100), 11)), axis=1)
#k = np.concatenate((k_s,k_s,k_s,k_s,k_s),axis=1)
#N = np.concatenate((np.full_like(k_s, 100),np.full_like(k_s, 200),np.full_like(k_s, 300),np.full_like(k_s, 400),np.full_like(k_s, 500)),axis=1)

energy = (1000*np.asarray(latency))*(np.asarray(power)/1000) # in mJoules

# Save results
file = open('EnergyPlot_GPU.pkl', 'wb')
pickle.dump(energy.tolist(), file)
pickle.dump(energyEst, file)
print("Data saved in file EnergyPlot_GPU.pkl")

# Plot results
plt.figure()
x = np.linspace(1, len(energy), len(energy))
plt.plot(x,np.asarray(energy))
plt.plot(x,np.asarray(energyEst))
plt.grid()
plt.show()
