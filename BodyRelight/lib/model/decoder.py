import torch
from torch import nn
from lib.model.Conv import *
from lib.model.DeConv import *
from .ResNetBlock import ResNetBlock

class Decoder(nn.Module):
    def __init__(self, output, clamp):
        """
        :param output: 9 for transport and 3 for albedo
        """
        super(Decoder, self).__init__()

        self.res_net = ResNetBlock()
        self.deconv = MultiDeConv([512, 512, 512, 256, 128, 64, output], clamp=clamp)
    
    def forward(self, x_list, x):
        '''
        :return : output of ResNetBlock and Deconv, the ResNetBlock result should be concatenated to output of encoder
        '''
        res_result = self.res_net(x)
        return res_result, self.deconv(x_list, res_result)