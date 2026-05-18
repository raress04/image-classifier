import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    Arhitectura MLP pentru clasificare.
    Presupunem o intrare redimensionata (ex. 128x128x3).
    """
    def __init__(self, num_classes=7, input_size=128*128*3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    """
    Arhitectura CNN comuna (max 4 conv, max 2 liniare).
    Include AdaptiveAvgPool2d pentru a fi agnostica la rezolutia intrarii.
    """
    def __init__(self, num_classes=7, first_kernel_size=3):
        super().__init__()
        # Calculam padding pentru a pastra dimensiunile unde e posibil
        pad = first_kernel_size // 2 
        
        self.features = nn.Sequential(
            # Strat 1
            nn.Conv2d(3, 32, kernel_size=first_kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Strat 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Strat 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Strat 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Pool adaptiv la final
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1) # aplatizare (B, 256, 1, 1) -> (B, 256)
        return self.classifier(x)
