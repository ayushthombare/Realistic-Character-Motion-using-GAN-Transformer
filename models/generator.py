import torch.nn as nn

class MotionGenerator(nn.Module):
    def __init__(self, dim=75):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        return self.net(x)
