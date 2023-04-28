from lib.options import *
from MODNet.app.matte import Matte
from BodyRelight.app.relight import Relight
import MHFormer.app.gen as gen
import numpy as np
# 预处理
opt = Options().parse() # 一些配置，比如batch_size，gpu_id等
relight = Relight(opt)
matte = Matte(opt)
sc = gen.MHFormer()

while True:
    # 从图片流中读取一帧
    ret, frame = sc.get_img()
    
    # 图片流结束
    if ret == False:
        break
    
    exr = [] # TODO: UE生成EXR图片，需要转换成sh
    
    light = np.asarray(exr)
    frame = np.asarray(frame)
    
    mask = matte.matte(frame)
    relighted_image = relight.relight(frame, mask, light, need_normalize = True)
    relighted_image = relighted_image.astype(np.uint8)
    key_points = sc.get_3D_kpt()
    
    # TODO: 把关键帧传给UE
