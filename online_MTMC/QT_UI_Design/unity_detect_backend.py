import torch
import cv2
import random
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from online_MTMC.utils.torch_utils import select_device
from online_MTMC.QT_UI_Design.models.experimental import attempt_load
from online_MTMC.utils.general import check_img_size, non_max_suppression, scale_coords
from online_MTMC.utils.datasets import LoadStreams_Ori, LoadImages_Ori, letterbox
from online_MTMC.utils.plots import plot_one_box

def load_detector_opt():
    parser = argparse.ArgumentParser()

    # Options for detection
    parser.add_argument('--det_name', type=str, default='yolov7-e6')
    parser.add_argument('--det_weights', type=str, default='C:/Users/cjh/Desktop/MTMCT/online_MTMC/preliminary/det_weights/')
    parser.add_argument('--img_size', type=int, default=[720, 1280], help='inference size (pixels)')
    parser.add_argument('--classes', type=int, default=[2, 5, 7], help='filter by class')
    parser.add_argument('--conf_thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--agnostic_nms', default=True, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')

    # Others
    parser.add_argument('--min_box_size', type=int, default=0.001, help='minimum box size')
    parser.add_argument('--img_ori_size', type=int, default=[1080, 1920], help='original image size (pixels)')
    parser.add_argument('--show_detection', default=True, action='store_true', help='display results')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--source', type=str, default='data/test/img.jpg', help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()

    return opt
def lodel_detector_model(opt):
    # 加载模型 ====================================================================================================
    # 加载检测模型
    det_model = attempt_load(opt.det_weights + opt.det_name + '.pt')
    det_model = det_model.cuda().eval().half()

    # 设置检测模型参数
    stride = int(det_model.stride.max())
    img_size = opt.img_size.copy()
    img_size[0] = check_img_size(opt.img_size[0], s=stride)
    img_size[1] = check_img_size(opt.img_size[1], s=stride)

    # 准备环境
    roi_masks, overlap_regions = {}, {}
    cams = ['drone1', 'drone2']
    for cam in cams:
        roi_masks[cam] = cv2.imread('C:/Users/cjh/Desktop/MTMCT/online_MTMC/QT_UI_Design/rois/%s.jpg' % cam,
                                    cv2.IMREAD_GRAYSCALE)
        overlap_regions[cam] = {}
        for cam_ in cams:
            overlap_regions[cam][cam_] = cv2.imread(
                'C:/Users/cjh/Desktop/MTMCT/online_MTMC/QT_UI_Design/overlap_zones/%s_%s.jpg' % (cam, cam_),
                cv2.IMREAD_GRAYSCALE) if cam_ != cam else None

    # 预热模型
    with torch.autocast('cuda'):
        for _ in range(10):
            det_model(torch.rand((4, 3, img_size[0], img_size[1]), device='cuda').half())

    # 是否显示检测结果
    is_show_detections = opt.show_detection
    # 选择处理设备
    device = select_device(opt.device)
    # 使用显卡处理时使用半精度
    half = device.type != 'cpu'
    # 获取检测器的检测类别
    names = det_model.module.names if hasattr(det_model, 'module') else det_model.names
    # 设置label
    label = 'running'

    return opt, stride, img_size, cams, roi_masks, det_model, device, half, names, is_show_detections, label

def main_detect_process(opt, stride, img_size, cams, roi_masks, det_model, device, half, names, is_show_detections, label, source_open):

    # 运行配置 ==============================================================================================================
    # 初始化
    img_h, img_w = opt.img_ori_size
    source = opt.source
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    source = source_open
    if webcam:
        cudnn.benchmark = True  # 加速图像推理
        dataset = LoadStreams_Ori(source, img_size=img_size, stride=stride)
    else:
        dataset = LoadImages_Ori(source, img_size=img_size, stride=stride)

    valid_cam = {}

    for path, img, im0s, vid_cap in dataset:
        batch_img = torch.zeros((len(cams), 3, img_size[0], img_size[1]), device='cuda').half()
        batch_img_ori = torch.zeros((len(cams), 3, opt.img_ori_size[0], opt.img_ori_size[1]), device='cuda').half()
        for cdx in range(len(cams)):
            batch_img[cdx] = torch.tensor(img / 255.0, device='cuda').half()
            batch_img_ori[cdx] = torch.tensor(im0s.transpose((2, 0, 1)) / 255.0, device='cuda').half()
        # 检测图像 =====================================================================================================
        with torch.autocast('cuda'):
            # batch_img 是输入的图像数据
            # args.augment 是指定是否对输入图像进行数据增强
            # preds 是推理的结果，通常包括了检测到的目标的类别、位置和置信度等信息
            # 通过使用 [0]，代码选择了输出列表的第一个元素，可能是最相关或最高置信度的目标信息
            preds = det_model(batch_img, augment=opt.augment)[0]

        # 非极大值抑制
        preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres,
                                    classes=opt.classes, agnostic=opt.agnostic_nms)

        det_count, detection = 0, {}
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        for pdx, pred in enumerate(preds):
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[pdx], '%g: ' % pdx, im0s[pdx].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            # 保存检测结果
            detection[cams[pdx]] = np.zeros((0, 5))

            # 当检测结果不为空时
            if len(pred) > 0:
                # 缩放检测框至原始图像尺寸
                pred[:, :4] = scale_coords(batch_img.shape[2:], pred[:, :4], batch_img_ori.shape[2:4])

                # 处理检测结果
                for *xyxy, conf, cls in reversed(pred):
                    # Convert to integer
                    x1, y1 = round(xyxy[0].item()), round(xyxy[1].item())
                    x2, y2 = round(xyxy[2].item()), round(xyxy[3].item())

                    # 根据检测框大小进行过滤
                    if (x2 - x1) * (y2 - y1) <= img_h * img_w * opt.min_box_size / 2:
                        continue

                    # 添加检测框
                    new_box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1), conf.item()])
                    new_box = new_box[np.newaxis, :]
                    detection[cams[pdx]] = np.concatenate([detection[cams[pdx]], new_box], axis=0)

                    det_count += 1

                    # 显示检测结果
                    if is_show_detections:
                        class_label = f'{names[int(cls)]} {conf:.2f}'
                        if pdx == 0:
                            plot_one_box(xyxy, im0, label=class_label, color=colors[int(cls)],
                                         line_thickness=opt.line_thickness)
                        if pdx == 1:
                            plot_one_box(xyxy, im0, label=class_label, color=colors[int(cls)],
                                         line_thickness=opt.line_thickness)

    return im0, label