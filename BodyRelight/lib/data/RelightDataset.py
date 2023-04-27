from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import torch

class RelightDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train', projection_mode='orthogonal',light_n=2,sample_light=True):
        self.opt = opt
        self.projection_mode = projection_mode

        # Path setup
        self.data_root = self.opt.dataroot
        self.metadata_root = os.path.join(self.opt.dataroot, '..', 'datas')
        self.LIGHT = os.path.join(self.metadata_root, 'sh')
        self.PARAM = os.path.join(self.metadata_root, 'PARAM')

        self.is_train = (phase == 'train')
        self.is_eval = (phase == 'eval')
        self.load_size = self.opt.loadSize

        self.subjects = self.get_subjects()
        self.lights=self.get_lights()
        self.light_n=light_n
        self.sample_light = sample_light

    def get_subjects(self):
        """
        Get the all the train/eval image files' names
        """
        all_subjects = os.listdir(self.data_root)
        var_subjects = np.loadtxt(os.path.join(self.metadata_root, 'val.txt'), dtype=str) # 测试用的数据集
        if len(var_subjects) == 0 and not self.is_eval:
            return all_subjects
        elif len(var_subjects) == 0 and self.is_eval:
            raise RuntimeError('No eval dataset')

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        elif self.is_eval:
            return sorted(list(var_subjects))
        else:
            raise RuntimeError('Unexpected phase')

    def get_lights(self):
        return np.load(os.path.join(self.LIGHT, os.listdir(self.LIGHT)[0]))

    def __len__(self):
        return len(self.subjects) * (self.light_n if self.sample_light and self.light_n < self.lights.shape[0] else self.lights.shape[0])

    def get_item(self, index):
        """
        Traverse all the images of one object in different lights first, then another object.
        """

        light_id = index % ( self.light_n if self.sample_light and self.light_n < self.lights.shape[0] else self.lights.shape[0] )
        # subject_id = index // self.lights.shape[0]
        subject_id = 0
        subject = self.subjects[subject_id]

        # Set up file path
        if(self.sample_light):
            light_id=0
            # light_id=np.random.randint(0,self.lights.shape[0])
        # image_path = os.path.join(self.data_root, '%04d' % (subject_id), 'IMAGE', '%04d.jpg' % (light_id))
        mask_path = os.path.join(self.data_root, '%04d' % (subject_id), 'MASK', 'MASK.png')
        albedo_path = os.path.join(self.data_root, '%04d' % (subject_id), 'ALBEDO', 'ALBEDO.jpg')
        transport_dir = os.path.join(self.data_root, '%04d' % (subject_id), 'TRANSFORM')

        # --------- Read groundtruth file data ------------
        # mask
        # [H, W] {0， 1.0}
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mask_3d = mask[:, :, None]

        # # image
        # # [H, W, 3]
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = image / 255.0
        # image = image * mask_3d

        # albedo
        # [H, W, 3] 0 ~ 1 float
        albedo = cv2.imread(albedo_path, cv2.IMREAD_COLOR) / 255.0
        albedo = albedo * mask_3d

        # light
        # [9, 3] SH coefficient
        light = self.lights[light_id, :, :]

        # transport
        # [H, W, 9]
        transport = []
        for i in range(9):
            transport_path = os.path.join(transport_dir, '%01d.jpg' % (i))
            tmp = cv2.imread(transport_path, cv2.IMREAD_GRAYSCALE) / 255.0
            if len(transport) == 0:
                transport = tmp[:, :, None]
            else:
                transport = np.concatenate((transport, tmp[:, :, None]), axis= 2)
        transport = transport * mask_3d
    
        mask = torch.from_numpy(mask.astype(np.float32))
        albedo = torch.from_numpy(albedo.astype(np.float32))
        light = torch.from_numpy(light.astype(np.float32))
        transport = torch.from_numpy(transport.astype(np.float32))

        image = albedo * torch.clamp(torch.matmul(transport, light), 0, 10.0)
        image = image * 2 - 1
        image = image.permute(2, 0, 1)

        albedo = albedo.permute(2, 0, 1)
        transport = transport.permute(2, 0, 1)

        res = {
            'name': subject,
            'image': image,         # [3, H, W]     [-1, 1]
            'mask': mask,           # [H, W]        {0, 1.0}
            'albedo': albedo,       # [3, H, W]     [0, 1.0]
            'light': light,         # [9, 3]        [-1.., +1..]
            'transport': transport  # [9, H, W]   
        }

        return res

    def __getitem__(self, index):
        return self.get_item(index)