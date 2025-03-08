# Simplified CNN for ASCAD (src/modeling/cnn.py)
class ASCAD_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=50, stride=5),  # Local pattern detection
            nn.ReLU(),
            nn.MaxPool1d(4),                             # Translation invariance
            nn.Conv1d(32, 64, kernel_size=25),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*13, 256),                       # Adapt based on trace length
            nn.ReLU(),
            nn.Linear(256, 256)                          # 256 classes per key byte
        )
