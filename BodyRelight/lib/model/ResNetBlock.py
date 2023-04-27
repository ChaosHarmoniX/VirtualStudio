import torch
from torch.nn.functional import relu
from torch import nn
from math import sqrt

class ResNetBlock(nn.Module):
    def __init__(self, input = 512, output = 512, kernel_size = 3, stride = 1, padding = 1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(input, output, kernel_size, stride, padding)
        self.batch_norm1 = nn.BatchNorm2d(output)
        self.conv2 = nn.Conv2d(output, output, kernel_size, stride, padding)
        self.batch_norm2 = nn.BatchNorm2d(output)

        nn.init.constant_(self.conv1.weight, sqrt(2))
        nn.init.constant_(self.conv2.weight, sqrt(2))
    
    def forward(self, x):
        y = relu(self.batch_norm1(self.conv1(x)))
        y = self.batch_norm2(self.conv2(y))
        return x + y