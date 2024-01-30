import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = Convolution(2, 3, 64)
        self.conv2 = Convolution(2, 64, 128)
        self.conv3 = Convolution(3, 128, 256)
        self.conv4 = Convolution(3, 256, 512)
        self.conv5 = Convolution(3, 512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.lin1 = Linear(512*7*7, 4096)
        self.lin2 = Linear(4096, 4096)
        self.lin3 = Linear(4096, n_classes, final=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x


class Convolution(nn.Module):
    def __init__(self, conv_count, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        layers = []
        for i in range(conv_count):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 
                                        kernel_size=kernel_size, stride=stride, padding=padding))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 
                                        kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, final=False):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.final = final

        if not self.final:
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.linear(x)
        if not self.final:
            x = self.activation(x)
            x = self.dropout(x)
        return x