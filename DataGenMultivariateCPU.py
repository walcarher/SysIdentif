import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Size of input tensor
#WH_in_list = [224, 32]
WH_in_list = np.linspace(11, 100, 10, dtype = int ).tolist()
#C_in_list = [3, 512]
C_in_list = np.linspace(3, 512, 10, dtype = int ).tolist()
# Vector list of multiple output tensor channels (number of filters)
#C_out_list = [32, 1024]
C_out_list = np.linspace(1, 512, 10, dtype = int ).tolist()
# Number of iterations to be averaged
n_iter = 10
# Delay between tests for memory synchronization
time_delay = 0.00
# Latency, power, energy and throughput vectors to store results
LAT = 0
POW = 0
E = 0
T = 0
dataset = []

# Forcing CPU use. Check with 'nvpmodel -q' before proceeding to use 4-core ARM and/or 2-core Denver CPU 
device = torch.device("cpu")

# This files are to be found in the L4T version of TX2, this may vary in the future
# It reads the latest CPU Power in mW from the INA3221 in the TegraTX2
cpuLoadFile = '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power1_input'


for WH_in in WH_in_list:
    for C_in in C_in_list:
        for C_out in C_out_list:
        
            # Convolutional layers definition with kernel size of kxk, stride 1, 0 padding, ReLU activation function and no pooling layer   
            class Conv1x1_s1_Net(nn.Module):

                def __init__(self):
                    super(Conv1x1_s1_Net, self).__init__()
                    # input channel, output channels, kernel size 1x1, stride 1-1, 0 padding
                    self.conv1 = nn.Conv2d(C_in, C_out, 1, stride = 1, padding = 0)

                def forward(self, x):
                    # Maxpooling 2x2 and ReLu activation function
                    #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
                    x = F.relu(self.conv1(x))
                    #x = self.conv1(x)
                    return x
                        
            class Conv3x3_s1_Net(nn.Module):

                def __init__(self):
                    super(Conv3x3_s1_Net, self).__init__()
                    # input channel, output channels, kernel size 3x3, stride 1-1, 0 padding
                    self.conv1 = nn.Conv2d(C_in, C_out, 3, stride = 1, padding = 0)

                def forward(self, x):
                    # Maxpooling 2x2 and ReLu activation function
                    #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
                    x = F.relu(self.conv1(x))
                    #x = self.conv1(x)
                    return x
                        
            class Conv5x5_s1_Net(nn.Module):

                def __init__(self):
                    super(Conv5x5_s1_Net, self).__init__()
                    # input channel, output channels, kernel size 7x7, stride 1-1, 0 padding
                    self.conv1 = nn.Conv2d(C_in, C_out, 5, stride = 1, padding = 0)

                def forward(self, x):
                    # Maxpooling 2x2 and ReLu activation function
                    #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
                    x = F.relu(self.conv1(x))
                    #x = self.conv1(x)
                    return x
                        
            class Conv7x7_s1_Net(nn.Module):

                def __init__(self):
                    super(Conv7x7_s1_Net, self).__init__()
                    # input channel, output channels, kernel size 7x7, stride 1-1, 0 padding
                    self.conv1 = nn.Conv2d(C_in, C_out, 7, stride = 1, padding = 0)

                def forward(self, x):
                    # Maxpooling 2x2 and ReLu activation function
                    #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
                    x = F.relu(self.conv1(x))
                    #x = self.conv1(x)
                    return x

            class Conv11x11_s1_Net(nn.Module):

                def __init__(self):
                    super(Conv11x11_s1_Net, self).__init__()
                    # input channel, output channels, kernel size 11x11, stride 1-1, 0 padding
                    self.conv1 = nn.Conv2d(C_in, C_out, 11, stride = 1, padding = 0)

                def forward(self, x):
                    # Maxpooling 2x2 and ReLu activation function
                    #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
                    x = F.relu(self.conv1(x))
                    #x = self.conv1(x)
                    return x

            convkxk_s1_net_list = [Conv1x1_s1_Net(), Conv3x3_s1_Net(), Conv5x5_s1_Net(), Conv7x7_s1_Net(), Conv11x11_s1_Net()]
            
            for convkxk_s1_net in convkxk_s1_net_list:
                      
                # Delay  
                time.sleep(time_delay)
                
                elapsed_time = 0.0
                iter = 0
                power = 0
                input = torch.rand(1, C_in, WH_in, WH_in, device = device)
                convkxk_s1_net.conv1.weight.data.random_()
                convkxk_s1_net.conv1.bias.data.random_()          
                out = convkxk_s1_net(input)              
                print("Input Tensor Size: %d Number of Channels: %d Filter size: %d  Number of Filters: %d"  % (WH_in, C_in, convkxk_s1_net.conv1.kernel_size[0], C_out))
                start = time.time()
                # Iterate over multiple tests
                while(iter < n_iter): 
                    input = torch.rand(1, C_in, WH_in, WH_in, device = device)
                    convkxk_s1_net.conv1.weight.data.random_()
                    convkxk_s1_net.conv1.bias.data.random_()           
                    out = convkxk_s1_net(input)
                    with open(cpuLoadFile, 'r') as cpuFile:
                        power += float(cpuFile.read())
                    iter += 1
                 
                elapsed_time = time.time() - start
        
                # Obtained results
                av_elapsed_time = elapsed_time/n_iter
                LAT = 1000*av_elapsed_time # in miliSeconds
                av_power = power/n_iter
                POW = av_power/1000 # in Watts
                E = POW * av_elapsed_time # in mJoules
                T = WH_in * WH_in * C_in * 4 / (av_elapsed_time * 1024 * 1024 * 1024) # in GBytes/seconds
                
                dataset.append([WH_in, C_in, convkxk_s1_net.conv1.kernel_size[0], C_out, LAT, POW, E, T])

# Save results
file = open('datasetMultivariateCPU.pkl', 'wb')
pickle.dump(dataset, file)
print("Data saved in file datasetMultivariateCPU.pkl")



