import sys
import os
import torch
from argparse import Namespace

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from lib.model.BodyRelightNet import BodyRelightNet
import numpy as np
import cv2
import torch.nn.functional as F

class Relight:
    def __init__(self, opt) -> None:
        # set cuda
        self.device = torch.device('cuda:%d' % opt.gpu_id)
        self.net = BodyRelightNet().to(device=self.device)
        
        self.net.load_state_dict(torch.load(opt.relight_model_path, map_location=self.device))
        self.net.eval()

    def relight_batch(self, image, mask, light, need_normalize = True):
        """
        Parameters:
        - image: `numpy.ndarray([batch, H, W, 3])`, default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - mask: `numpy.ndarray([H, W])`, default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - light: `numpy.ndarray([batch, 9, 3])`
        - need_normalize: if True, the input image will be divided by 255 to be normalized to [0, 1] when relighting

        Return:
        - religted_image: `numpy.ndarray([batch, H, W, 3])`, ranging [0, 255]
        """
        with torch.no_grad():
            image = torch.from_numpy(image).float()
            light = torch.from_numpy(light).float()
            mask = torch.from_numpy(mask).float()

            image = image.to(self.device)
            light = light.to(self.device)
            mask = mask.to(self.device)

            if need_normalize:
                image = image / 255.0
                mask = mask / 255.0
            image = image * mask[:, :, :, None]

            image = 2 * image - 1 # normalize to [-1, 1]
            
            image = image.permute(0, 3, 1, 2) # [batch, H, W, 3] -> [batch, 3, H, W]

            # resize to 1024 * 1024
            original_shape = image.shape
            if image.shape[2] != 1024 or image.shape[3] != 1024:
                image = F.interpolate(image, size=(1024, 1024), mode='area')

            albedo_eval, light_eval, transport_eval = self.net(image)
            light = light.permute(0, 2, 1) # [batch, 9, 3] -> [batch, 3, 9]
            shading = torch.clamp(torch.einsum('ijk,iklm->ijlm', light, transport_eval), 0.0, 10.0) # [batch, 3, 1024, 1024]
            image_eval = torch.clamp(albedo_eval * shading, 0.0, 1.0) * 255.0
            
            # resize to original size
            if image_eval.shape[2] != original_shape[2] or image_eval.shape[3] != original_shape[3]:
                image_eval = F.interpolate(image_eval, size=(original_shape[2], original_shape[3]), mode='area')

            image_eval = image_eval.permute(0, 2, 3, 1) # [batch, 3, H, W] -> [batch, H, W, 3]
            return image_eval.to('cpu').numpy()
    
    def relight(self, image, mask, light, need_normalize = True):
        """
        Parameters:
        - image: `numpy.ndarray([H, W, 3])`, default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - mask: `numpy.ndarray([H, W])`, default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - light: `numpy.ndarray([9, 3])`
        - need_normalize: if True, the input image will be divided by 255 to be normalized to [0, 1] when relighting

        Return:
        - religted_image: `numpy.ndarray([H, W, 3])`, ranging [0, 255]
        """
        with torch.no_grad():
            image = torch.from_numpy(image).float()
            light = torch.from_numpy(light).float()
            mask = torch.from_numpy(mask).float()

            image = image.to(self.device)
            light = light.to(self.device)
            mask = mask.to(self.device)

            if need_normalize:
                image = image / 255.0
                mask = mask / 255.0
            image = image * mask[:, :, None]
            
            image = 2 * image - 1 # normalize to [-1, 1]
            image = image.permute(2, 0, 1) # [H, W, 3] -> [3, H, W]
            
            image.unsqueeze_(dim = 0)
            
            # resize to 1024 * 1024
            original_shape = image.shape
            if image.shape[2] != 1024 or image.shape[3] != 1024:
                image = F.interpolate(image, size=(1024, 1024), mode='area')
            
            albedo_eval, light_eval, transport_eval = self.net(image)

            albedo_eval.squeeze_(dim = 0)
            transport_eval.squeeze_(dim = 0)
            transport_eval = transport_eval.permute(1, 2, 0) # [9, 1024, 1024] -> [1024, 1024, 9]
            shading = torch.clamp(torch.matmul(transport_eval, light), 0.0, 10.0)
            shading = shading.permute(2, 0, 1) # [1024, 1024, 3] -> [3, 1024, 1024]
            image_eval = torch.clamp(albedo_eval * shading, 0.0, 1.0) * 255.0

            # resize to original size
            if image_eval.shape[1] != original_shape[2] or image_eval.shape[2] != original_shape[3]:
                image_eval = F.interpolate(image_eval.unsqueeze(dim=0), size=(original_shape[2], original_shape[3]), mode='area')
                image_eval.squeeze_(dim = 0)

            image_eval = image_eval.permute(1, 2, 0) # [3, H, W] -> [H, W, 3]
            return image_eval.to('cpu').numpy()
    

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..')))
    from lib.options import Options

    opt = Options().parse()
    relight = Relight(opt)
    image = cv2.imread('./BodyRelight/eval/IMAGE.jpg', cv2.IMREAD_COLOR)
    mask = cv2.imread('./BodyRelight/eval/MASK.png', cv2.IMREAD_GRAYSCALE)
    light = np.load('./BodyRelight/datas/LIGHT/env_sh.npy')[0]

    # --- test non-batch version ---
    # image_eval = relight.relight(image, mask, light)
    # cv2.imwrite('./BodyRelight/eval/IMAGE_eval.jpg', image_eval)
    # # Don't use cv2.imshow directly, it can't deal with float32 ranging from 0 to 255 properly
    # image_eval = image_eval.astype(np.uint8)
    # cv2.imshow('image_eval', image_eval)
    # cv2.waitKey(0)

    # --- test batch version ---
    image = image[None, :, :, :]
    mask = mask[None, :, :]
    light = light[None, :, :]
    image_eval = relight.relight_batch(image, mask, light)
    image_eval = image_eval[0]
    cv2.imwrite('./BodyRelight/eval/IMAGE_L_eval.jpg', image_eval)
    # Don't use cv2.imshow directly, it can't deal with float32 ranging from 0 to 255 properly
    image_eval = image_eval.astype(np.uint8)
    cv2.imshow('image_eval', image_eval)
    cv2.waitKey(0)