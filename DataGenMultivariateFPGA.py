import os
import csv
import pickle
import numpy as np

# Data generation parser from Quartus Synthesis VHDL dataset
# Dataset is stored as [WH_in, C_in, k, C_out]
dataset = []
# KPI dataset [LAT, POW, E, T, R_ALM, R_ALUT, R_LAB, R_M20K]
kpi_dataset = []
# Datapath to post synthesis generated VHDL from PyTorch-ONNX to DeliRium and finally passed through Quartus Pro Edition 17.1 
directory = 'vhdl_generated_synthesis'


# Iterate over files in directory
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.csv'):
            # File name is splitted in convkxk, HW, C, N
            splittedFilename = filename.split('_')
            WH_in = int(splittedFilename[1]) # Getting WH
            C_in = int(splittedFilename[2]) # Getting C_in
            k = int(splittedFilename[0].split('x')[-1]) # Getting k
            C_out = int(splittedFilename[3]) # Getting C_out or N
            if not [WH_in, C_in, k, C_out] in dataset: # New element in the list
                dataset.append([WH_in, C_in, k, C_out])
                # Latency, power, energy, throughput and resources KPIs
                LAT, POW, E, T, R_ALM, R_ALUT, R_LAB, R_M20K = 0, 0 ,0 ,0 ,0 ,0 , 0, 0
                kpi_dataset.append([LAT, POW, E, T, R_ALM, R_ALUT, R_LAB, R_M20K])
            # Open file to fill KPI's synthetized information    
            with open(os.path.join(root, filename), newline = '') as csvfile:
                reader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
                for i, row in enumerate(reader):
                    if i == 7 and  splittedFilename[6][:-4] == 'power': # POW in W
                        POW = float(row[1].split(' ')[0])/1000 # POW in W
                        kpi_dataset[dataset.index([WH_in, C_in, k, C_out])][1] = POW
                    if i == 1 and splittedFilename[6][:-4] == 'fmax': # fmax in MHz for LAT and T
                        LAT = WH_in*WH_in*C_in*C_out/(1000*float(row[0].split(' ')[0])) # LAT in ms
                        T = (1 / LAT)/1000 # T in GB/s
                        kpi_dataset[dataset.index([WH_in, C_in, k, C_out])][0] = LAT
                        kpi_dataset[dataset.index([WH_in, C_in, k, C_out])][3] = T
                    if splittedFilename[6][:-4] == 'resources': # Resources
                        if i == 2: # ALMs 
                            R_ALM = int(row[1].replace(',',''))
                            kpi_dataset[dataset.index([WH_in, C_in, k, C_out])][4] = R_ALM
                        if i == 21: # ALUTs
                            R_ALUT = int(row[1].replace(',',''))
                            kpi_dataset[dataset.index([WH_in, C_in, k, C_out])][5] = R_ALUT
                        if i == 17: # LABs
                            R_LAB = int(row[1].split(' ')[0].replace(',',''))
                            kpi_dataset[dataset.index([WH_in, C_in, k, C_out])][6] = R_LAB
                        if i == 44: # M20Ks (issue colliding with 'I/O pins')
                            if(row[0] == 'M20K blocks'):
                                R_M20K = int(row[1].split(' ')[0].replace(',',''))
                                kpi_dataset[dataset.index([WH_in, C_in, k, C_out])][7] = R_M20K 
                        if i == 48: # M20Ks (issue colliding with 'I/O pins')
                            if(row[0] == 'M20K blocks'):
                                R_M20K = int(row[1].split(' ')[0].replace(',',''))
                                kpi_dataset[dataset.index([WH_in, C_in, k, C_out])][7] = R_M20K

# Initialize full dataset
full_dataset = []
# Compute Energy as E = LAT * POW for each element
for feature_sample, kpi_sample in zip(dataset, kpi_dataset):
    kpi_sample[2] = kpi_sample[0] * kpi_sample[1] / 1000 # in Joules
    # Concatenate feature and KPI datset lists
    full_dataset.append(feature_sample + kpi_sample)

# Save results
file = open('datasetMultivariateFPGA.pkl', 'wb')
pickle.dump(full_dataset, file)
print("Data saved in file datasetMultivariateFPGA.pkl")



