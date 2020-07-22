import sys
import time
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
WH_in = 224
C_in = 3
# Vector list of multiple output tensor channels (number of filters)
#C_out_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
C_out_list = np.linspace(2, 1024, 20, dtype = int ).tolist()
# Number of iterations to be averaged
n_iter = 200
# Delay between tests for synchronization en memory 
time_delay = 0.1
# Latency vector to store results
MMACs_list = []
LAT_list = []

# Run test over C_out sizes
for C_out in C_out_list:
    
    print("Number of output channels: %d" % C_out)
    
    class Conv1x1_s1_Net(nn.Module):

        def __init__(self):
            super(Conv1x1_s1_Net, self).__init__()
            # input channel, output channels, kernel size 1x1, stride 1-1, 0 padding
            self.conv1 = nn.Conv2d(3, C_out, 1 , stride = 1, padding = 0).cuda()

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
    # Iterate over multiple tests
    while(iter < n_iter):
        torch.cuda.seed()
        input = torch.rand(1, C_in, WH_in, WH_in).cuda()
        conv1x1_s1_net.conv1.weight.data.random_().cuda()
        conv1x1_s1_net.conv1.bias.data.random_().cuda()
        start = time.time()
        out = conv1x1_s1_net(input).cuda()
        torch.cuda.synchronize()
        end = time.time()
        elapsed_time += end - start
        iter += 1
        
    av_elapsed_time = elapsed_time/n_iter
    LAT_list.append(1000*av_elapsed_time)
    torch.cuda.empty_cache()
    
    # Function that maps Conv to MACs k*k*Cin*Cout*Hout*Wout
    MACs = conv1x1_s1_net.conv1.kernel_size[0] * \
                conv1x1_s1_net.conv1.kernel_size[1] * \
                C_in * out.size(1) * out.size(2) * out.size(3)
                
    print("Number of MegaMACs on Conv1x1 with stride of 1: %.2f" %(MACs/1000000))
    
    MMACs_list.append(MACs/1000000)

# Pop first sample since GPU it is biased by initialization 
MMACs_list.pop(0)
LAT_list.pop(0) 
    
# Save results
file = open('dataset.pkl', 'wb')
pickle.dump([MMACs_list, LAT_list], file)
print("Data saved in file dataset.pkl")

L = len(LAT_list)
t = np.linspace(1, L, L, dtype = int ).tolist()

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
plt.scatter(MMACs_list, LAT_list)
plt.title('MACs vs Latency', fontsize = 20)
plt.xlabel('#MACs (Million)', fontsize = 18)
plt.ylabel('Latency (ms)', fontsize = 18)
plt.grid()

plt.show()


