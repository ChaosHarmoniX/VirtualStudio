import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
from argparse import Namespace

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.models.modnet import MODNet

class Matte:
    def __init__(self, opt: Namespace) -> None:
        self.net = MODNet(backbone_pretrained=False)
        self.net = nn.DataParallel(self.net)
        
        self.device = torch.device('cuda:%d' % opt.gpu_id)
        self.net = self.net.to(device=self.device)
        self.net.load_state_dict(torch.load(opt.matte_model_path, map_location=self.device))
        self.net.eval()
    
    def matte_batch(self, image, need_normalize = True):
        """
        Parameters:
        - image: `numpy.ndarray([batch, 1024, 1024, 3])`, default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - need_normalize: if True, the input image will be divided by 255 to be normalized to [0, 1] when relighting

        Return:
        - matte: `numpy.ndarray([batch, 1024, 1024, 3])`, ranging [0, 255]
        """
        with torch.no_grad():
            image = torch.from_numpy(image).float()
            image = image.to(self.device)

            if need_normalize:
                image = image / 255
            image = 2 * image - 1

            image = image.permute(0, 3, 1, 2) # [batch, 1024, 1024, 3] -> [batch, 3, 1024, 1024]
            # 在512下infer效果更佳
            image = F.interpolate(image, size=(512, 512), mode='area')
            _, _, matte = self.net(image, True)
            matte = F.interpolate(matte, size=(1024, 1024), mode='area')
            matte = matte * 255.0
            matte.squeeze_(dim=1)

            return matte.to('cpu').numpy().astype(np.uint8)

    def matte(self, image, need_normalize = True):
        """
        Parameters:
        - image: `numpy.ndarray([1024, 1024, 3])`, default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - need_normalize: if True, the input image will be divided by 255 to be normalized to [0, 1] when relighting

        Return:
        - matte: `numpy.ndarray([1024, 1024, 3])`, ranging [0, 255]
        """
        with torch.no_grad():
            image = torch.from_numpy(image).float()
            image = image.to(self.device)

            if need_normalize:
                image = image / 255
            image = 2 * image - 1

            image = image.permute(2, 0, 1) # [1024, 1024, 3] -> [3, 1024, 1024]
            image.unsqueeze_(dim=0)
            # 在512下infer效果更佳
            image = F.interpolate(image, size=(512, 512), mode='area')
            _, _, matte = self.net(image, True)
            matte = F.interpolate(matte, size=(1024, 1024), mode='area')
            matte.squeeze_(dim=0)
            matte.squeeze_(dim=0)
            matte = matte * 255.0

            return matte.to('cpu').numpy().astype(np.uint8)



if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..')))
    from lib.options import Options
    
    matte = Matte(Options().parse())
    image = cv2.imread('./MODNet/input/0000.jpg', cv2.IMREAD_COLOR) / 255.

    # --- test non batch ---
    result = matte.matte(image, False)
    cv2.imshow('test', result)
    cv2.imwrite('./MODNet/output/0000.jpg', result)
    cv2.waitKey(0)

    # --- test batch ---
    # image = image[None, :, :, :]
    # result = matte.matte_batch(image, False)
    # cv2.imshow('test', result[0])
    # cv2.imwrite('./MODNet/output/0000.jpg', result[0])
    # cv2.waitKey(0)