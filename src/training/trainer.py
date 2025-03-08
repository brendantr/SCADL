from src.modeling.cnn import CNNModel
from src.data_loading.dataset_splitter import split_dataset
from src.data_loading.data_loader import create_data_loaders

def train(model, device, train_loader, optimizer, criterion):
    # Training loop logic here

def main():
    # Load and split dataset
    dataset = load_dataset()
    train_dataset, val_dataset = split_dataset(dataset)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

    # Initialize model and optimizer
    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train(model, device, train_loader, optimizer, criterion)

if __name__ == "__main__":
    main()
