import torch
from torch import nn
from torch.nn import functional

class MultiConv(nn.Module):
    image_size = 128
    def __init__(self, filter_channels, kernel_size = 4, stride = 2, padding = 1, start_with_bn = False):
        super(MultiConv, self).__init__()
        self.filters = []
        self.batch_norms = []
        self.activate_func = functional.leaky_relu
        self.start_with_bn = start_with_bn

        for l in range(0, len(filter_channels) - 1):
            conv = nn.Conv2d(filter_channels[l], filter_channels[l + 1], kernel_size=kernel_size, stride=stride, padding=padding)
            nn.init.normal_(conv.weight, 0.0, 0.02)
            self.filters.append(conv)
            self.add_module("conv%d" % l, self.filters[l])

            if not ((l == 0 and start_with_bn == False) or (l == len(filter_channels) - 2 and start_with_bn)):
                self.batch_norms.append(nn.BatchNorm2d(filter_channels[l + 1]))
                self.add_module('batch_norm%d' % (len(self.batch_norms) - 1), self.batch_norms[-1])
        

    def forward(self, image):
        y = image
        results = []
        count = 0
        for i, f in enumerate(self.filters):
            y = f(y)
            if not ((i == 0 and self.start_with_bn == False) or (i == len(self.filters) - 1 and self.start_with_bn)):
                # batch normalization
                y = self.batch_norms[count](y)
                count += 1
            # 激活函数
            if not (i == len(self.filters) - 1 and self.start_with_bn):
                y = self.activate_func(y)
            results.append(y)
        return results[:-1:], results[-1]
