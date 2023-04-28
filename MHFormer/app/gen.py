import sys
import os 
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import argparse
import cv2
from demo.lib.preprocess import h36m_coco_format, revise_kpts
from demo.lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed

sys.path.append(os.getcwd())
from model.mhformer import Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)

# 输入视频路径，返回关键点
def sc_gen_2d_kpt(video_path):
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    return keypoints
    
# 模型初始化
def sc_model_init(model_path='checkpoint/mhformer/351'):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 351
    args.pad = (args.frames - 1) // 2
    args.previous_dir = model_path
    args.n_joints, args.out_joints = 17, 17
     ## Reload 
    model = Model(args).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model of MHFormer in 'checkpoint/pretrained/351'
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)

    model.eval()
    return model,args
    
# 输入视频路径，返回视频对象和视频总帧数    
def sc_get_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, video_length

# 输入视频对象，返回视频的下一帧图像
def sc_get_seq_image(cap):
    return cap.read()

# 输入视频对象和帧数，返回视频的指定帧图像
def sc_get_video_frame(cap,frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    return cap.read()

# 输入图像，返回图像的关节点坐标
def sc_get_3D_kpt( img, keypoints, model, frame_number,args):
    i=frame_number
    ret, img = img
    img_size = img.shape
    ## input frames
    start = max(0, i - args.pad)
    end =  min(i + args.pad, len(keypoints[0])-1)
    input_2D_no = keypoints[0][start:end+1]
        
    left_pad, right_pad = 0, 0
    if input_2D_no.shape[0] != args.frames:
        if i < args.pad:
            left_pad = args.pad - i
        if i > len(keypoints[0]) - args.pad - 1:
            right_pad = i + args.pad - (len(keypoints[0]) - 1)

        input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
    joints_left =  [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  
    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[ :, :, 0] *= -1
    input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
    input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
    
    input_2D = input_2D[np.newaxis, :, :, :, :]
    input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
    N = input_2D.size(0)
    ## estimation
    output_3D_non_flip = model(input_2D[:, 0])
    output_3D_flip     = model(input_2D[:, 1])
    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    output_3D = output_3D[0:, args.pad].unsqueeze(1) 
    output_3D[:, :, 0, :] = 0
    post_out = output_3D[0, 0].cpu().detach().numpy()
    rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    post_out = camera_to_world(post_out, R=rot, t=0)
    post_out[:, 2] -= np.min(post_out[:, 2])
    return post_out

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
parser.add_argument('--gpu', type=str, default='0', help='input video')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print("Using GPU: {} ".format(args.gpu)," --- if CUDA is available: ",torch.cuda.is_available())
video_path = './MHFormer/demo/video/' + args.video
video_name = video_path.split('/')[-1].split('.')[0]
output_dir = './MHFormer/demo/output/' + video_name + '/'
class MHFormer():
    def __init__(self) :
        self.sc_model,self.sc_args=sc_model_init()
        self.sc_video, self.sc_numberOfFrame = sc_get_video(video_path)
        self.sc_keypoints = sc_gen_2d_kpt(video_path)
        self.sc_frame_num = 0
        
        
    
    def get_3D_kpt(self):
        sc_imageOfFrame = sc_get_seq_image(self.sc_video)
        sc_out_3d_kpt = sc_get_3D_kpt(img=sc_imageOfFrame,
                                      keypoints=self.sc_keypoints,
                                      model=self.sc_model,
                                      frame_number=self.sc_frame_num,
                                      args=self.sc_args)
        self.sc_frame_num+=1
        print(sc_out_3d_kpt)
        return sc_out_3d_kpt
        
        
        
    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    # parser.add_argument('--gpu', type=str, default='0', help='input video')
    # args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # print("Using GPU: {} ".format(args.gpu)," --- if CUDA is available: ",torch.cuda.is_available())

    # video_path = './MHFormer/demo/video/' + args.video
    # video_name = video_path.split('/')[-1].split('.')[0]
    # output_dir = './MHFormer/demo/output/' + video_name + '/'

    sc_model,sc_args = sc_model_init()
    sc_video, sc_numberOfFrame = sc_get_video(video_path)
    sc_keypoints = sc_gen_2d_kpt(video_path)
    
    
    # 使用示例
    for i in range(sc_numberOfFrame):
        sc_imageOfFrame = sc_get_seq_image(sc_video)
        sc_out_3d_kpt = sc_get_3D_kpt(img=sc_imageOfFrame,
                                      keypoints=sc_keypoints,
                                      model=sc_model,
                                      frame_number=i,
                                      args=sc_args)
        print(sc_out_3d_kpt)
        # # plot it and save figures
        # plt.figure(figsize=(10, 10))
        # gs=gridspec.GridSpec(1,2)
        # gs.update(wspace=-0.00, hspace=0.05) 
        # ax = plt.subplot(gs[0], projection='3d')
        # show3Dpose(sc_out_3d_kpt, ax)
        # output_dir_3D = output_dir +'pose3D/'
        # os.makedirs(output_dir_3D, exist_ok=True)
        # plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
  
    
    print('Generating demo successful!')


