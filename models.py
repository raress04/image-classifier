import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    Arhitectura MLP pentru clasificare.
    Presupunem o intrare redimensionata (ex. 128x128x3).
    """
    def __init__(self, num_classes=7, input_size=128*128*3, dropout_prob: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    """
    Arhitectura CNN comuna (max 4 conv, max 2 liniare).
    Include AdaptiveAvgPool2d pentru a fi agnostica la rezolutia intrarii.
    """
    def __init__(self, num_classes=7, first_kernel_size=3, use_batchnorm=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        pad = first_kernel_size // 2

        def conv_block(in_ch, out_ch, kernel=3, padding=1, pool=False):
            """Construieste un bloc Conv -> [BN] -> ReLU -> [Pool]."""
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                          padding=padding, bias=not use_batchnorm)
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers

        feature_layers = [
            *conv_block(3,   32,  kernel=first_kernel_size, padding=pad, pool=True),
            *conv_block(32,  64,  pool=True),
            *conv_block(64,  128, pool=True),
            *conv_block(128, 256, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        ]
        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)  # (B, 256, 1, 1) -> (B, 256)
        return self.classifier(x)
