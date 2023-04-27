from turtle import forward
from lib.model.Conv import *
import torch
from torch import nn

from .decoder import Decoder

class BodyRelightNet(nn.Module):
    def __init__(self):
        super(BodyRelightNet, self).__init__()

        self.encoder = MultiConv(filter_channels=[3, 64, 128, 256, 512, 512, 512], start_with_bn=False)

        self.albedo_decoder = Decoder(output=3, clamp=False)
        self.transport_decoder = Decoder(output=9, clamp=True)
        self.light_decoder = MultiConv(filter_channels=[1536, 512, 256, 128, 27], start_with_bn=True)

    def forward(self, x):
        """
        :param x: [C_3, H, W]
        :return albedo_map, light_map, transport_map
        :albedo_map: [C_3, H, W]
        :light_map: [3, 9]
        :transport_map: [9, H, W]
        """
        fs, feature = self.encoder(x)
        
        albedo_feature, albedo_map = self.albedo_decoder(fs, feature)
        transport_feature, transport_map = self.transport_decoder(fs, feature)

        compose_feature = torch.cat((feature, albedo_feature, transport_feature), 1)
        _, light_map = self.light_decoder(compose_feature)
        light_map = torch.reshape(light_map, (-1, 9, 3))

        return albedo_map, light_map, transport_map # TODO: 可能顺序要变

