import h5py
import torch
from torch.utils.data import Dataset

class ASCADDataset(Dataset):
    def __init__(self, h5_path='data/raw/ASCAD.h5', train=True):
        self.h5 = h5py.File(h5_path, 'r')
        group = self.h5['Profiling_traces'] if train else self.h5['Attack_traces']
        self.traces = group['traces']
        self.labels = group['labels']
        
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = torch.tensor(self.traces[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return trace, label

