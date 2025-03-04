import torch
from torch.utils.data import DataLoader
from src.data_loading.ascad_loader import ASCADDataset  # New Line
from modeling.baseline_mlp import SideChannelMLP        # New path

def main():
    # Initialize dataset
    train_dataset = ASCADDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = SideChannelMLP(input_size=700, hidden_size=128, num_classes=256)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(5):  # We'll do 5 epochs as a test
        total_loss = 0
        for batch_idx, (traces, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Complete | Avg Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()

