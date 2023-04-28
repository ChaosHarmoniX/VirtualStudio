import argparse
import os


class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        src_path = "E:/MyProject/UE/Python/VirtualStudio/"
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--video_path', type=str, default = src_path + 'BodyRelight/datas/video/test.mp4')
        parser.add_argument('--buffer_time', type=float, default=10)
        parser.add_argument('--fps', type=float, default=0.01)

        g_matte = parser.add_argument_group('matte')
        g_matte.add_argument('--matte_model_path', type=str, default = src_path + 'MODNet/pretrained/modnet_webcam_portrait_matting.ckpt')

        g_relight = parser.add_argument_group('relight')
        g_relight.add_argument('--relight_model_path', type=str, default = src_path + 'BodyRelight/checkpoints/example/net_latest')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
