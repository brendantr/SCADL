import torch
from torch.utils.data import DataLoader
from data_loading.ascad_loader import ASCADDataset
from cpa_analysis import cpa_attack
import h5py
import os

# Evaluate the neural network model on the test dataset
test_dataset = ASCADDataset(dataset_path='data/raw/ASCAD.h5', train=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = torch.load('path/to/your/trained_model.pth')  # Load your trained model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

nn_accuracy = correct / total
print(f"Neural Network Test Accuracy: {nn_accuracy * 100:.2f}%")

# Perform CPA attack on the test dataset
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'ASCAD.h5'))
print(f"Loading dataset from: {file_path}")

with h5py.File(file_path, 'r') as f:
    traces = f['Attack_traces']['traces'][:]
    plaintexts = f['Attack_traces']['plaintext'][:]

num_traces = 1000  # Number of traces to use for the attack
correlations = cpa_attack(traces, plaintexts, num_traces)

# Identify the key hypothesis with the highest correlation
best_guess = np.argmax(np.max(correlations, axis=1))
print(f"Best key guess: {best_guess}")

# Compare CPA and Neural Network results
print(f"Neural Network Accuracy: {nn_accuracy * 100:.2f}%")
print(f"CPA Best Key Guess: {best_guess}")