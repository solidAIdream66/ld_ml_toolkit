import torch
import torch.nn as nn
from ..configs import ModelConfig
from torchvision.models import resnet18


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 stride: int = 1,
                 dropout: float = 0.0,
                 use_batch_norm: bool = True):
        super().__init__()
        self.dropout = dropout
        self.stride = stride
        bias = not use_batch_norm
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        
        if stride != 1 or in_channels != out_channels:
            downsample_layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            ]
            if use_batch_norm:
                downsample_layers.append(nn.BatchNorm2d(out_channels))
            self.downsample = nn.Sequential(*downsample_layers)
        else:
            self.downsample = nn.Identity()        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)
        out += identity

        out = self.relu(out)
        out = self.dropout(out)

        return out
    

class MyResidualNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        num_classes = config.num_classes
        dropout = config.dropout
        use_batch_norm = config.use_batch_norm
        channels = config.channels
        blocks_per_stage = config.blocks_per_stage

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(channels[0]) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        prev_channels = channels[0]
        for index, (channel, blocks) in enumerate(zip(channels, blocks_per_stage)):
            stride = 1 if index == 0 else 2
            layer = self._make_layer(prev_channels, channel, blocks, stride, dropout, use_batch_norm)
            self.layers.append(layer)
            prev_channels = channel

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], config.num_classes),
        )


    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout, use_batch_norm):
        layers = []
        layers.append(
            ResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                dropout=dropout,
                use_batch_norm=bool(use_batch_norm),
            )
        )

        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    dropout=dropout,
                    use_batch_norm=bool(use_batch_norm),
                )
            )
        
        return nn.Sequential(*layers)
               

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for layer in self.layers:
            x = layer(x)       

        x = self.head(x)

        return x

   

######################################################################################
#                                     TESTS
######################################################################################
# import pytest


# @pytest.fixture(scope="session")
# def data_loaders():
#     from .data import get_data_loaders

#     return get_data_loaders(batch_size=2)


# def test_model_construction(data_loaders):

#     model = MyModel(num_classes=23, dropout=0.3)

#     dataiter = iter(data_loaders["train"])
#     images, labels = next(dataiter)

#     out = model(images)

#     assert isinstance(
#         out, torch.Tensor
#     ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

#     assert out.shape == torch.Size(
#         [2, 23]
#     ), f"Expected an output tensor of size (2, 23), got {out.shape}"
