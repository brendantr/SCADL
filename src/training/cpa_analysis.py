import numpy as np
import h5py
from scipy.stats import pearsonr
import os

def hamming_weight(x):
    return bin(x).count('1')

def cpa_attack(traces, plaintexts, num_traces, num_hypotheses=256):
    num_samples = traces.shape[1]
    correlations = np.zeros((num_hypotheses, num_samples))

    for k in range(num_hypotheses):
        hypothetical_power = np.array([hamming_weight(p ^ k) for p in plaintexts[:num_traces]])
        for t in range(num_samples):
            actual_power = traces[:num_traces, t]
            correlations[k, t] = pearsonr(hypothetical_power, actual_power)[0]

    return correlations

# Load the dataset
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'ASCAD.h5'))
print(f"Loading dataset from: {file_path}")

with h5py.File(file_path, 'r') as f:
    traces = f['Profiling_traces']['traces'][:]
    plaintexts = f['Profiling_traces']['plaintext'][:]

# Perform CPA attack
num_traces = 1000  # Number of traces to use for the attack
correlations = cpa_attack(traces, plaintexts, num_traces)

# Identify the key hypothesis with the highest correlation
best_guess = np.argmax(np.max(correlations, axis=1))
print(f"Best key guess: {best_guess}")