import torch.nn as nn
from .configs import ModelConfig






class ConvClassifier(nn.Module):
    """VGG-style CNN classifier with a modern pooling head.

    With channels=(64, 128, 256, 512, 512) and blocks_per_stage=2 this
    matches VGG-13 (arXiv:1409.1556). The original 3-layer FC head is
    replaced by global average pooling + one hidden layer, reducing
    parameter count while maintaining performance.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        layers = []
        in_channels = 3

        for out_channels in config.channels:
            for _ in range(config.blocks_per_stage):
                layers.append(nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=1,
                    bias=not config.use_batch_norm,
                ))
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(config.channels[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)
