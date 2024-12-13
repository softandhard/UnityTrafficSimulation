import argparse


class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Options for detection
        self.parser.add_argument('--det_name', type=str, default='yolov7-e6')
        self.parser.add_argument('--det_weights', type=str, default='./preliminary/det_weights/')
        self.parser.add_argument('--img_size', type=int, default=[720, 1280], help='inference size (pixels)')
        self.parser.add_argument('--classes', type=int, default=[2, 5, 7], help='filter by class')
        self.parser.add_argument('--conf_thres', type=float, default=0.1, help='object confidence threshold')
        self.parser.add_argument('--iou_thres', type=float, default=0.7, help='IoU threshold for NMS')
        self.parser.add_argument('--agnostic_nms', default=True, action='store_true', help='class-agnostic NMS')
        self.parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')

        # Options for feature extraction
        self.parser.add_argument('--feat_ext_name', type=str, default='resnet50_ibn_a')
        self.parser.add_argument('--scene_feat_ext_name', type=str, default='resnext101_ibn_a')
        self.parser.add_argument('--feat_ext_weights', type=str, default='./preliminary/feat_ext_weights/')
        self.parser.add_argument('--scene_feat_ext_weights', type=str, default='./preliminary/feat_ext_weights/')
        self.parser.add_argument('--avg_type', type=str, default='gap')
        self.parser.add_argument('--scene_avg_type', type=str, default='gap')
        self.parser.add_argument('--patch_size', type=int, default=[384, 384], help='inference size (pixels)')
        self.parser.add_argument('--part_patch_size', type=int, default=[128, 128], help='inference part-img size (pixels)')
        self.parser.add_argument('--num_class', type=int, default=960)

        # Options for MTSC
        self.parser.add_argument("--det_high_thresh", type=float, default=0.6)
        self.parser.add_argument("--det_low_thresh", type=float, default=0.1)
        self.parser.add_argument("--cos_thr", type=float, default=0.6)
        self.parser.add_argument("--iou_thr", type=float, default=0.6)
        self.parser.add_argument("--max_time_lost", type=int, default=30)

        # Options for MTMC
        self.parser.add_argument('--get_feat_mode', type=str, default='best')
        self.parser.add_argument("--max_time_differ", type=int, default=30)
        self.parser.add_argument("--mtmc_match_thr", type=float, default=0.65)

        # Others
        self.parser.add_argument('--data_dir', type=str, default='D:/Users/ddd/VeRi/AIC22/test/S02/')
        self.parser.add_argument('--output_dir', type=str, default='./outputs/')
        self.parser.add_argument('--min_box_size', type=int, default=0.001, help='minimum box size')
        self.parser.add_argument('--img_ori_size', type=int, default=[1080, 1920], help='original image size (pixels)')
        self.parser.add_argument('--merge_pattern', type=str, default='minimal', help='minimal / average')
        self.parser.add_argument('--cluster_merge_pattern', type=str, default='average', help='minimal / average')
        self.parser.add_argument('--is_merge', type=str, default=True, help='whether to merge the results')

    def parse(self):
        return self.parser.parse_args()


opt = Opts().parse()
