import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle

# CUDA device system to be identified (Energy)
if torch.cuda.is_available():
    print("Found ", torch.cuda.device_count(), "CUDA GPU device(s)")
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    print("Initializing first CUDA device: ", torch.cuda.get_device_name(device))
    torch.cuda.init()
else :
    sys.exit("No CUDA device was found to be identified. Exiting program...")

# Size of input tensor
WH_in_list = [224, 32]
#WH_in_list = np.linspace(7, 224, 20, dtype = int ).tolist()
C_in_list = [3, 128]
#C_in_list = np.linspace(3, 1024, 20, dtype = int ).tolist()
# Vector list of multiple output tensor channels (number of filters)
C_out_list = [32, 128]
#C_out_list = np.linspace(1, 1024, 20, dtype = int ).tolist()
# Number of samples 
n_samples = 250
t = np.linspace(1, n_samples-1, n_samples-1, dtype = int ).tolist()
# Number of iterations to be averaged
n_iter = 200
# Delay between tests for memory synchronization
time_delay = 0.00
# Latency vector to store results
MMACs_list = []
LAT_list = []
POW_list = []

# This files are to be found in the L4T version of TX2, this may vary in the future
# It reads the latest GPU Power in mW from the INA3221 in the TegraTX2
gpuLoadFile = '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power0_input'

# Run test over different random W_in H_in C_in and C_out sizes from a white noise distribution
n = 0
while n < n_samples:

    #Random values for the PRBS  
    if len(WH_in_list) == len(C_in_list) and len(WH_in_list) == len(C_out_list) and len(C_in_list) == len(C_out_list):
        bin = random.randint(0,len(WH_in_list)-1)
        WH_in = WH_in_list[bin]
        C_in = C_in_list[bin]
        C_out = C_out_list[bin] 
    # Random values for the PRBS with different lengths 
    else:
        WH_in = WH_in_list[random.randint(0,len(WH_in_list)-1)] 
        C_in = C_in_list[random.randint(0,len(C_in_list)-1)] 
        C_out = C_out_list[random.randint(0,len(C_out_list)-1)]
    
    print("Number of output channels: %d" % C_out)
    
    class Conv1x1_s1_Net(nn.Module):

        def __init__(self):
            super(Conv1x1_s1_Net, self).__init__()
            # input channel, output channels, kernel size 1x1, stride 1-1, 0 padding
            self.conv1 = nn.Conv2d(C_in, C_out, 1 , stride = 1, padding = 0).cuda()

        def forward(self, x):
            # Maxpooling 2x2 and ReLu activation function
            #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
            x = F.relu(self.conv1(x)).cuda()
            #x = self.conv1(x).cuda()
            return x
        
    conv1x1_s1_net = Conv1x1_s1_Net()
    
    print(conv1x1_s1_net)
    time.sleep(time_delay)
    
    elapsed_time = 0.0
    iter = 0
    power = 0
    # Iterate over multiple tests
    while(iter < n_iter):
        torch.cuda.seed()
        input = torch.rand(1, C_in, WH_in, WH_in).cuda()
        conv1x1_s1_net.conv1.weight.data.random_().cuda()
        conv1x1_s1_net.conv1.bias.data.random_().cuda()
        start = time.time()
        out = conv1x1_s1_net(input).cuda()
        with open(gpuLoadFile, 'r') as gpuFile:
            power += float(gpuFile.read())
        torch.cuda.synchronize()
        end = time.time()
        elapsed_time += end - start
        iter += 1
        
    av_elapsed_time = elapsed_time/n_iter
    LAT_list.append(1000*av_elapsed_time) # in miliseconds
    av_power = power/n_iter
    POW_list.append(av_power/1000) # in Watts
    torch.cuda.empty_cache()
    
    # Function that maps Conv to MACs = k*k*Cin*Cout*Hout*Wout
    MACs = conv1x1_s1_net.conv1.kernel_size[0] * \
                conv1x1_s1_net.conv1.kernel_size[1] * \
                C_in * out.size(1) * out.size(2) * out.size(3)
                
    print("Number of MegaMACs on Conv1x1 with stride of 1: %.2f" %(MACs/1000000))
    
    MMACs_list.append(MACs/1000000)
    
    n += 1 

# Pop first sample since GPU it is biased by initialization 
MMACs_list.pop(0)
LAT_list.pop(0)
POW_list.pop(0) 
# Save results
file = open('datasetPRBS.pkl', 'wb')
pickle.dump([MMACs_list, LAT_list, POW_list], file)
print("Data saved in file datasetPRBS.pkl")

# Show results
plt.figure()
plt.plot(t, MMACs_list )
plt.title('MACs vs iteration', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('#MACs (Million)', fontsize = 18)
plt.grid()

plt.figure()
plt.plot(t, LAT_list)
plt.title('Latency vs iteration', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Latency (ms)', fontsize = 18)
plt.grid()

plt.figure()
plt.plot(t, POW_list)
plt.title('Power vs iteration', fontsize = 20)
plt.xlabel('Iteration', fontsize = 18)
plt.ylabel('Power (W)', fontsize = 18)
plt.grid()

plt.figure()
plt.scatter(MMACs_list, LAT_list)
plt.title('MACs vs Latency', fontsize = 20)
plt.xlabel('#MACs (Million)', fontsize = 18)
plt.ylabel('Latency (ms)', fontsize = 18)
plt.grid()

plt.show()



