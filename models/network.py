import torch.nn as nn
from utils.init_weights import init_weights

class CustomModel(nn.Module):
    def __init__(self, init_type='xavier'):
        """
        Example model architecture with weight initialization.

        Args:
            init_type: Weight initialization method ('normal', 'xavier', 'kaiming', or 'orthogonal')
        """
        super().__init__()
        
        # Example architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )
        
        # Initialize weights using the function from init_weights.py
        self.apply(init_weights(init_type))
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 