import torch
import os
import random
import glob
from torch.utils.data import Dataset
import numpy as np

class StateMeasurementResultData(Dataset):
    
    def __init__(self,num_observables=242,num_states=50):
        observables = []
        for i in range(0,num_observables):
            tmp = np.load('/home/haoyu/5qubit/float_observable5_'+str(i)+'.npy')
            observables.append(np.array(tmp))
        self.observables = np.array(observables)
        values = []
        for k in range(0,9):    
            base_path = '/home/haoyu/Data6/qubits=5/layers=30/t_gate=' + str(k)
            file_pattern = base_path + '/circuit_seed=*/measuredQubits=01234/*.npy'
            file_paths = glob.glob(file_pattern)
            for file_path in file_paths:
                tmp = np.load(file_path)
                tmp = tmp.reshape(-1, 2**5)
                values.append(np.array(tmp, dtype=np.float32))
        self.expectation_values = np.array(values)
    def __getitem__(self, idx):
        assert idx < len(self.expectation_values)
        return self.observables, self.expectation_values[idx]
    
    def __len__(self):
        return len(self.expectation_values)
class TestStateMeasurementResultData(Dataset):
    def __init__(self,num_observables=27,num_states=10):
        observables = []
        for i in range(0,num_observables):
            tmp = np.load('/home/haoyu/5qubit/float_observable5_'+str(i)+'.npy')
            observables.append(np.array(tmp))
        self.observables = np.array(observables)
        
        ratio_space = np.linspace(-2,2,41)
        values = []
        for k in range(0,0):    
            base_path = '/home/haoyu/Data6/qubits=5/layers=30/t_gate=' + str(k)
            file_pattern = base_path + '/circuit_seed=*/measuredQubits=01234/*.npy'
            file_paths = glob.glob(file_pattern)
            for file_path in file_paths:
                tmp = np.load(file_path)
                tmp = tmp.reshape(-1, 2**5)
                values.append(np.array(tmp, dtype=np.float32))        
        self.expectation_values = np.array(values)


    def __getitem__(self, idx):
        assert idx < len(self.expectation_values)
        return self.observables, self.expectation_values[idx]

    def __len__(self):
        return len(self.expectation_values)


