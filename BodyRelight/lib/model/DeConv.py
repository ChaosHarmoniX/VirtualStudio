import torch
from torch import nn
from torch.nn import functional

class MultiDeConv(nn.Module):
    def __init__(self, filter_channels, kernel_size = 4, stride = 2, padding = 1, clamp = False):
        super(MultiDeConv, self).__init__()
        self.filters = []
        self.batch_norms = []
        self.clamp = clamp

        for l in range(0, len(filter_channels) - 1):
            deconv = nn.ConvTranspose2d(filter_channels[l] if l == 0 else filter_channels[l] * 2, filter_channels[l + 1], kernel_size=kernel_size, stride=stride, padding=padding)
            nn.init.normal_(deconv.weight, 0.0, 0.02)
            self.filters.append(deconv)
            self.add_module("deconv%d" % l, self.filters[l])
            
            if l != len(filter_channels) - 2:
                self.batch_norms.append(nn.BatchNorm2d(filter_channels[l + 1]))
                self.add_module('batch_norm%d' % (len(self.batch_norms) - 1), self.batch_norms[-1])
        

    def forward(self, x_list, res_result):
        y = res_result
        for i, f in enumerate(self.filters):
            y = f(y)
            if i != len(self.filters) - 1:
                # batch normalization
                y = self.batch_norms[i](y)
            if i < 3:
                y = functional.dropout(y, 0.5, training=True)
            # 激活函数
            if i != len(self.filters) - 1:
                y = functional.relu(y)
                y = torch.cat((y, x_list[-i-1]), 1)
        if self.clamp:
            torch.clamp(y, 0.0, 1.0)
        return y
