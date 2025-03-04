import h5py
import torch
from torch.utils.data import Dataset
import os

class ASCADDataset(Dataset):
    def __init__(self, dataset_path='data/raw/ASCAD.h5', train=True):
        # Load the dataset
        self.data = self.load_data(dataset_path, train)

    def load_data(self, dataset_path, train):
        # Update the file path to the correct location of your ASCAD dataset
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', dataset_path))
        print(f"Loading dataset from: {file_path}")  # Print the absolute path for debugging
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with h5py.File(file_path, 'r') as f:
            if train:
                data = f['Profiling_traces']['traces'][:]
                labels = f['Profiling_traces']['labels'][:]
            else:
                data = f['Attack_traces']['traces'][:]
                labels = f['Attack_traces']['labels'][:]
        return data, labels

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        trace = torch.tensor(self.data[0][idx], dtype=torch.float32)
        label = torch.tensor(self.data[1][idx], dtype=torch.long)
        return trace, label