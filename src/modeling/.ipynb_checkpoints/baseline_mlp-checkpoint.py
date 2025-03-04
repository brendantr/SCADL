import torch.nn as nn

class SideChannelMLP(nn.Module):
    def __init__(self, input_size=700, hidden_size=128, num_classes=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)

