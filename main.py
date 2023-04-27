import cv2
from lib.options import *
from BodyRelight.app.relight import Relight
import numpy as np
from time import sleep
from threading import Thread

from MODNet.app.matte import Matte

# TODO: 可能加入list中的都为一个对象

if __name__ == '__main__':
    opt = Options().parse() # 一些配置，比如batch_size，gpu_id等
    print('\n'.join(['{0}: {1}'.format(item[0], item[1]) for item in opt.__dict__.items()])) # 打印所有配置
    relight = Relight(opt)
    matte = Matte(opt)
    
    video = cv2.VideoCapture(opt.video_path)

    # tmp light
    light = np.load('./BodyRelight/datas/LIGHT/env_sh.npy')[0]

    relight_frames = []

    # check image size
    video.grab()
    _, frame = video.retrieve()
    assert frame.shape[0] == frame.shape[1], 'image size must be square'
    
    # def play():
    #     sleep(opt.buffer_time)
    #     while len(relight_frames) > 0:
    #         frame_to_play = relight_frames.pop(0)
    #         cv2.imshow('relight', frame_to_play)
    #         sleep(1 / opt.fps)

    # # 在另一个线程中播放视频流
    # t = Thread(target=play)
    # t.start()

    # 抠图并重打光，将处理结果放入relight_frames队列中
    if opt.batch_size == 1:
        while True:
            ret, frame = video.retrieve() # 读取一帧图片
            if ret == False:
                print('Done')
                break

            mask = matte.matte(frame)
            relighted_image = relight.relight(frame, mask, light, need_normalize = True)
            relighted_image = relighted_image.astype(np.uint8)
            relight_frames.append(relighted_image)
            # cv2.imshow('image_eval', relighted_image)

            # test picture
            # frame = cv2.imread('./MODNet/input/0000.jpg', cv2.IMREAD_COLOR)
            # mask = matte.matte(frame)
            # relighted_image = relight.relight(frame, mask, light, need_normalize = True)
            # relighted_image = relighted_image.astype(np.uint8)
            # cv2.imshow('image_eval', relighted_image)
            # cv2.waitKey(0)

            video.grab()
    else:
        while True:
            count = 0
            while count < opt.batch_size:
                ret, frame = video.retrieve() # 读取一帧图片
                if ret == False:
                    print('Done')
                    break
                if count == 0:
                    frames = frame[None, :, :, :]
                else:
                    frames = np.stack([frames, frame[None, :, :, :]], axis = 0)
                count += 1

                video.grab()

            masks = matte.matte_batch(frames)
            relighted_images = relight.relight_batch(frames, masks, light, need_normalize = True)
            relighted_images = relighted_images.astype(np.uint8)
            for i in range(frames.shape[0]):
                relight_frames.append(relighted_images[i])
                # cv2.imshow('image_eval', relighted_images[i])
                # cv2.waitKey(1)

            if ret == False:
                break