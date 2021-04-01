import os
import csv
import pickle

# Data generation parser from Quartus Synthesis VHDL dataset

# Latency, power, energy, throughput and resources KPIs
LAT = 0
POW = 0
E = 0
T = 0
R_ALM = 0
R_ALUT = 0
R_LAB = 0
R_M20K = 0
datasetArray = np.zeros((10, 10, 2, 10), dtype=float32) # with lengths of HW, C, k, N
dataset = []
# Datapath to post synthesis generated VHDL from PyTorch-ONNX to DeliRium and finally passed through Quartus Pro Edition 17.1 
directory = 'vhdl_generated_synthesis'


# Iterate over files in directory
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.csv'):
            print(filename)
            # File name is splitted in convkxk, HW, C, N
            splittedFilename = filename.split('_')
            with open(os.path.join(root, filename), newline = '') as csvfile:
                reader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
                for i, row in enumerate(reader):
                    if i == 7 and  splittedFilename[6][:-4] == 'power': # POW in mW
                        print(row)
                    if i == 1 and splittedFilename[6][:-4] == 'fmax': # fmax in MHz for LAT and T
                        print(row)
                    if i == 2 and splittedFilename[6][:-4] == 'resources': # ALMs 
                        print(row)
                    if i == 21 and splittedFilename[6][:-4] == 'resources': # ALUTs
                        print(row)
                    if i == 17 and splittedFilename[6][:-4] == 'resources': # LABs
                        print(row)
                    if i == 48 and splittedFilename[6][:-4] == 'resources': # M20Ks
                        print(row)
#dataset.append([WH_in, C_in, convkxk_s1_net.conv1.kernel_size[0], C_out, LAT, POW, E, T])

print(dataset)

# Save results
#file = open('datasetMultivariateFPGA.pkl', 'wb')
#pickle.dump(dataset, file)
#print("Data saved in file datasetMultivariateFPGA.pkl")



