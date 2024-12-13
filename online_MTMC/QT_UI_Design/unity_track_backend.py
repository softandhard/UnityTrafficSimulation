import argparse
import random
import socket
import time
import copy
import cv2
import torch
import os
import numpy as np

from online_MTMC.QT_UI_Design.unity_opts import opt
from torchvision import transforms
from online_MTMC.utils.sklearn_dunn import dunn
from online_MTMC.scene_tracking.bot_sort import BoTSORT
from online_MTMC.QT_UI_Design.models.experimental import attempt_load
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from online_MTMC.models.feature_extractor import FeatureExtractor
from online_MTMC.models.feature_extractor_deepsort_CBAM import FeatureExtractor as FeatureExtractor_Deepsort
from online_MTMC.utils.scipy_linear_assignment import linear_assignment
from online_MTMC.utils.general import check_img_size, non_max_suppression, scale_coords
from online_MTMC.utils.utils import letterbox, letterbox_part, class_agnostic_nms, pairwise_tracks_dist
from online_MTMC.utils.datasets import letterbox as convert_letterbox
from online_MTMC.utils.torch_utils import select_device
from online_MTMC.utils.plots import plot_one_box

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(2000)]
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, box, identitie=None, offset=(0, 0)):
    cx, cy, w, h = [int(i) for i in box]
    cx += offset[0]
    cy += offset[0]
    w += offset[1]
    h += offset[1]
    x1 = cx
    y1 = cy
    x2 = x1 + w
    y2 = y1 + h
    # box text and bar
    id = int(identitie) if identitie is not None else 0
    color = compute_color_for_labels(id)
    label = '{}{:d}'.format("", id)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(
        img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    cv2.putText(img, label, (x1, y1 +
                             t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img

class Cluster:
    def __init__(self):
        super(Cluster, self).__init__()
        self.tracks = []
        self.feat = np.zeros((0, 2048))

    def add_track(self, track):
        if track not in self.tracks:
            self.tracks.append(track)

    def get_feature(self):
        if opt.get_feat_mode != 'all':
            feat = np.array([track.get_feature(opt.get_feat_mode) for track in self.tracks])
            feat = feat[np.newaxis, :] if len(feat.shape) == 1 else feat
        else:
            feat = np.concatenate([track.get_feature(opt.get_feat_mode) for track in self.tracks], axis=0)
        return feat

    def get_scene_feature(self):
        if opt.get_feat_mode != 'all':
            scene_feat = np.array([track.get_scene_feature(opt.get_feat_mode) for track in self.tracks])
            scene_feat = scene_feat[np.newaxis, :] if len(scene_feat.shape) == 1 else scene_feat
        else:
            scene_feat = np.concatenate([track.get_scene_feature(opt.get_feat_mode) for track in self.tracks], axis=0)
        return scene_feat

    @ property
    def end_frame(self):
        return np.max([track.end_frame for track in self.tracks])

    @ property
    def cam_list(self):
        return [track.cam for track in self.tracks]

# 创建一个与Unity引擎通信的Socket连接
def create_unity_connection():
    # 创建Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 服务端IP和端口
    server_ip = '127.0.0.1'
    server_port = 11111

    # 绑定IP和端口
    server_socket.bind((server_ip, server_port))

    # 监听
    server_socket.listen(1)

    print(f"服务器正在监听 {server_ip}:{server_port}")

    # 接受连接
    client_socket, client_address = server_socket.accept()

    print(f"IP地址 {client_address} 的连接已经建立.")

    return client_socket

# 通过客户端套接字接收两幅图像数据
def receive_unity_data(client_socket):
    img_t1 = time.time()

    # 接收图像大小
    size_data = client_socket.recv(4)  # 假设图像大小以4字节整数形式发送

    image_size = int.from_bytes(size_data, byteorder='little')

    # 接收图像数据
    image_data = b''
    while len(image_data) < image_size:
        UAV1_packet = client_socket.recv(image_size - len(image_data))
        if not UAV1_packet:
            break
        image_data += UAV1_packet

    # 接收图像大小
    size_data2 = client_socket.recv(4)  # 假设图像大小以4字节整数形式发送

    image_size2 = int.from_bytes(size_data2, byteorder='little')

    # 接收图像数据
    image_data2 = b''
    while len(image_data2) < image_size2:
        UAV2_packet = client_socket.recv(image_size2 - len(image_data2))
        if not UAV2_packet:
            break
        image_data2 += UAV2_packet

    # 将接收到的数据转换为numpy数组
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    # 从numpy数组中恢复图像
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # 将接收到的数据转换为numpy数组
    image_array2 = np.frombuffer(image_data2, dtype=np.uint8)
    # 从numpy数组中恢复图像
    image2 = cv2.imdecode(image_array2, cv2.IMREAD_COLOR)

    img_t3 = time.time()

    print("unity-FPS==", 1 / (img_t3 - img_t1))

    return image, image2

# 预处理图像
def preprocess_img(cam, drone1_img, drone2_img, img_size, stride):
    img = None
    img_ori = None

    if cam == 'drone1':
        # Padded resize
        img1 = convert_letterbox(drone1_img, img_size, stride=stride)[0]
        # Convert
        img1 = img1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img1 = np.ascontiguousarray(img1)
        img = img1
        img_ori = drone1_img

    if cam == 'drone2':
        # Padded resize
        img2 = convert_letterbox(drone2_img, img_size, stride=stride)[0]
        # Convert
        img2 = img2[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img2 = np.ascontiguousarray(img2)

        img = img2
        img_ori = drone2_img


    return img, img_ori

# 解析命令行参数，并返回解析后的参数值
def load_trackor_opt():
    parser = argparse.ArgumentParser()

    # Options for detection
    parser.add_argument('--det_name', type=str, default='yolov7-w6')
    parser.add_argument('--det_weights', type=str,
                             default='C:/Users/cjh/Desktop/MTMCT/online_MTMC/preliminary/det_weights/')
    parser.add_argument('--img_size', type=int, default=[720, 1280], help='inference size (pixels)')
    parser.add_argument('--classes', type=int, default=[2, 5, 7], help='filter by class')
    parser.add_argument('--conf_thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--agnostic_nms', default=True, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')

    # Options for feature extraction
    parser.add_argument('--feat_ext_name', type=str, default='resnet50_ibn_a')
    parser.add_argument('--scene_feat_ext_name', type=str, default='attention_net')
    parser.add_argument('--feat_ext_weights', type=str,
                             default='C:/Users/cjh/Desktop/MTMCT/online_MTMC/preliminary/feat_ext_weights/')
    parser.add_argument('--scene_feat_ext_weights', type=str,
                             default='C:/Users/cjh/Desktop/MTMCT/online_MTMC/preliminary/feat_ext_weights/')
    parser.add_argument('--avg_type', type=str, default='gap')
    parser.add_argument('--scene_avg_type', type=str, default='gap')
    parser.add_argument('--patch_size', type=int, default=[384, 384], help='inference size (pixels)')
    parser.add_argument('--part_patch_size', type=int, default=[128, 128], help='inference part-img size (pixels)')
    parser.add_argument('--num_class', type=int, default=960)

    # Options for MTSC
    parser.add_argument("--det_high_thresh", type=float, default=0.6)
    parser.add_argument("--det_low_thresh", type=float, default=0.1)
    parser.add_argument("--cos_thr", type=float, default=0.6)
    parser.add_argument("--iou_thr", type=float, default=0.6)
    parser.add_argument("--max_time_lost", type=int, default=30)

    # Options for MTMC
    parser.add_argument('--get_feat_mode', type=str, default='best')
    parser.add_argument("--max_time_differ", type=int, default=10)
    parser.add_argument("--mtmc_match_thr", type=float, default=0.65)

    # Others
    parser.add_argument('--data_dir', type=str, default='D:/Users/ddd/VeRi/AIC22/test/S02/')
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--min_box_size', type=int, default=0.001, help='minimum box size')
    parser.add_argument('--img_ori_size', type=int, default=[1080, 1920], help='original image size (pixels)')
    parser.add_argument('--merge_pattern', type=str, default='minimal', help='minimal / average')
    parser.add_argument('--cluster_merge_pattern', type=str, default='minimal')
    parser.add_argument('--show_img', default=False, help='是否显示仿真图像')
    parser.add_argument('--show_detection', default=True, action='store_true', help='display results')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels)')
    args = parser.parse_args()

    return args

# 加载并初始化Yolov7和Bortsort模型，用于对象检测和跟踪
def load_model(args):
    # 加载模型 ====================================================================================================
    # 加载检测模型
    det_model = attempt_load(args.det_weights + args.det_name + '.pt')
    det_model = det_model.cuda().eval().half()

    # 设置检测模型参数
    stride = int(det_model.stride.max())
    img_size = args.img_size.copy()
    img_size[0] = check_img_size(args.img_size[0], s=stride)
    img_size[1] = check_img_size(args.img_size[1], s=stride)

    # 加载特征提取模型
    feat_ext_model = FeatureExtractor(args.feat_ext_name, args.avg_type, args.feat_ext_weights)
    feat_ext_model = feat_ext_model.cuda().eval().half()

    # 加载特征提取模型
    scene_feat_ext_model = FeatureExtractor_Deepsort(args.scene_feat_ext_name, args.scene_avg_type,
                                                     args.scene_feat_ext_weights)
    scene_feat_ext_model = scene_feat_ext_model.cuda().eval().half()
    # 归一化处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 准备环境 ========================================================================================================
    # 其他准备处理
    datasets, trackers, f_nums = {}, {}, []
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

        # 准备追踪器
        print("cam==", cam)
        trackers[cam] = BoTSORT(args)

    # 预热模型
    with torch.autocast('cuda'):
        for _ in range(10):
            det_model(torch.rand((4, 3, img_size[0], img_size[1]), device='cuda').half())
            feat_ext_model(torch.rand((10, 3, args.patch_size[0], args.patch_size[1]), device='cuda').half())
            scene_feat_ext_model(
                torch.rand((10, 3, args.part_patch_size[0], args.part_patch_size[1]), device='cuda').half())

    # 是否显示检测结果
    is_show_detections = args.show_detection
    # 选择处理设备
    device = select_device(args.device)
    # 使用显卡处理时使用半精度
    half = device.type != 'cpu'
    # 根据view_img的值来确定is_show_track的值
    is_show_tracks = False if is_show_detections else True
    # 获取检测器的检测类别
    names = det_model.module.names if hasattr(det_model, 'module') else det_model.names
    # 设置label
    label = 'running'

    return args, stride, img_size, normalize, cams, roi_masks, overlap_regions, det_model, feat_ext_model, scene_feat_ext_model, \
        trackers, device, half, names, is_show_detections, is_show_tracks, label

def main_track_process(args, client_socket, stride, img_size, normalize, cams, roi_masks, overlap_regions, det_model, feat_ext_model,
                 scene_feat_ext_model, trackers, device, half, names, is_show_detections, is_show_tracks, label):

    drone1_img, drone2_img = receive_unity_data(client_socket)

    # 运行配置 ==============================================================================================================
    # 初始化
    img_h, img_w = args.img_ori_size
    next_global_id, dunn_index_prev = 0, -1e5
    clusters_dict = {}
    fdx = 0
    track_fsp = 0
    detect_fps = 0
    detect1_info = " "
    detect2_info = ""
    track1_info = ""
    track2_info = ""
    if drone1_img is not None and drone2_img is not None:

        # 生成空batch
        batch_img = torch.zeros((len(cams), 3, img_size[0], img_size[1]), device='cuda').half()
        batch_img_ori = torch.zeros((len(cams), 3, args.img_ori_size[0], args.img_ori_size[1]), device='cuda').half()
        batch_patch = torch.zeros((100 * len(cams), 3, args.patch_size[0], args.patch_size[1]), device='cuda').half()
        batch_part_patch = torch.zeros((100 * len(cams), 3, args.part_patch_size[0], args.part_patch_size[1]), device='cuda').half()
        batch_img_cam = {}

        # 准备图像
        valid_cam = {}
        for cdx, cam in enumerate(cams):
            # 读取图像
            valid_cam[cam] = True
            drone1_img_resize = cv2.resize(drone1_img, (1920, 1080))
            drone2_img_resize = cv2.resize(drone2_img, (1920, 1080))
            drone_img_new, drone_img_ori = preprocess_img(cam, drone1_img_resize, drone2_img_resize, img_size, stride)
            batch_img_cam[cam] = drone_img_ori
            # 检查是否为空
            if drone_img_new is None:
                valid_cam[cam] = False
                continue

            # Store
            batch_img[cdx] = torch.tensor(drone_img_new / 255.0, device='cuda').half()
            batch_img_ori[cdx] = torch.tensor(drone_img_ori.transpose((2, 0, 1)) / 255.0, device='cuda').half()

        start_time = time.time()
        detect_start = time.time()
        # 检测图像 =====================================================================================================
        with torch.autocast('cuda'):
            # batch_img 是输入的图像数据
            # args.augment 是指定是否对输入图像进行数据增强
            # preds 是推理的结果，通常包括了检测到的目标的类别、位置和置信度等信息
            # 通过使用 [0]，代码选择了输出列表的第一个元素，可能是最相关或最高置信度的目标信息
            preds = det_model(batch_img[list(valid_cam.values())], augment=args.augment)[0]

        # 非极大值抑制
        preds = non_max_suppression(preds, args.conf_thres, args.iou_thres,
                                    classes=args.classes, agnostic=args.agnostic_nms)

        # 如果图像为空，插入空张量
        for cdx, cam in enumerate(cams):
            if not valid_cam[cam]:
                preds.insert(cdx, torch.zeros((0, 6)).cuda().half())

        detect_end = time.time()
        detect_time = detect_end - detect_start
        if detect_time != 0:
            detect_fps = 1 / detect_time
            print("检测帧率", detect_fps)

        # 准备提取特征 =================================================================================
        # 准备特征提取模型
        det_count, detection = 0, {}
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        for pdx, pred in enumerate(preds):
            # 保存检测结果
            detection[cams[pdx]] = np.zeros((0, 5))

            # 当检测结果不为空时
            if len(pred) > 0:
                # 缩放检测框至原始图像尺寸
                pred[:, :4] = scale_coords(batch_img.shape[2:], pred[:, :4], batch_img_ori.shape[2:4])

                # 获取检测结果
                if pdx == 0:
                    for c in pred[:, -1].unique():
                        n = (pred[:, -1] == c).sum()  # 检测每个类别
                        detect1_info += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 加入检测结果
                        track1_info += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 加入追踪结果

                if pdx == 1:
                    for c in pred[:, -1].unique():
                        n = (pred[:, -1] == c).sum()  # 检查每个类别
                        detect2_info += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 加入检测结果
                        track2_info += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 加入追踪结果

                # 处理检测结果
                for *xyxy, conf, cls in reversed(pred):
                    # Convert to integer
                    x1, y1 = round(xyxy[0].item()), round(xyxy[1].item())
                    x2, y2 = round(xyxy[2].item()), round(xyxy[3].item())

                    # Filter detections with RoI mask
                    if roi_masks[cams[pdx]][min(y2 + 1, img_h) - 1, (max(x1, 0) + min(x2 + 1, img_w)) // 2] == 0:
                        continue

                    # Filter detection with box size
                    if (x2 - x1) * (y2 - y1) <= img_h * img_w * args.min_box_size / 2:
                        continue

                    # Add detections
                    new_box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1), conf.item()])
                    new_box = new_box[np.newaxis, :]
                    detection[cams[pdx]] = np.concatenate([detection[cams[pdx]], new_box], axis=0)

                    # Get patch
                    # patch 原始大小为目标检测所获取的图像大小
                    patch = batch_img_ori[pdx][:, max(y1, 0):min(y2 + 1, img_h), max(x1, 0):min(x2 + 1, img_w)]
                    patch = normalize(letterbox(patch))
                    # 获取目标框的部分图像
                    part_patch = batch_img_ori[pdx][:, max(y1, 0):min(y2 + 1, img_h), max(x1, 0):min(x2 + 1, img_w)]
                    part_patch = normalize(letterbox_part(part_patch))

                    # 缩放为指定分辨率 patch [1, 3, 384, 384]
                    batch_patch[det_count] = torch.fliplr(patch) if cams[pdx] == 'c008' else patch
                    # 缩放为指定分辨率 [1, 3, 128, 128]
                    batch_part_patch[det_count] = torch.fliplr(part_patch) if cams[pdx] == 'c008' else part_patch
                    det_count += 1

                    # 显示检测结果
                    if is_show_detections:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if pdx == 0:
                            plot_one_box(xyxy, drone1_img, label=label, color=colors[int(cls)], line_thickness=args.line_thickness)
                        if pdx == 1:
                            plot_one_box(xyxy, drone2_img, label=label, color=colors[int(cls)], line_thickness=args.line_thickness)

        ext_start = time.time()
        # 提取特征
        with torch.autocast('cuda'):
            batch_patch = batch_patch[:det_count]
            batch_feat = feat_ext_model(batch_patch)
        batch_feat = batch_feat.squeeze().cpu().detach().numpy()
        # 提取特征
        with torch.autocast('cuda'):
            batch_part_patch = batch_part_patch[:det_count]
            batch_scene_feat = scene_feat_ext_model(batch_part_patch)
        batch_scene_feat = batch_scene_feat.squeeze().cpu().detach().numpy()

        ext_end = time.time()
        ext_time = ext_end - ext_start
        print("特征提取时间：", ext_time)

        # 单相机追踪 ========================================================================
        # 分离特征
        feat_count, feat = 0, {}
        scene_count, scene_feat = 0, {}
        for cam in cams:
            feat[cam] = batch_feat[feat_count:feat_count + len(detection[cam])]
            feat_count += len(detection[cam])

            scene_feat[cam] = batch_scene_feat[scene_count:scene_count + len(detection[cam])]
            scene_count += len(detection[cam])

        # 获取在线轨迹
        online_tracks_raw = {}
        for cam in cams:
            online_tracks_raw[cam] = trackers[cam].update(cam, detection[cam], feat[cam], scene_feat[cam])

        # 准备多相机多目标追踪环境 =================================================================
        # 过滤轨迹
        online_tracks_filtered = {}
        for cam in cams:
            online_tracks_filtered[cam] = []
            for track in online_tracks_raw[cam]:
                # If not activated
                if not track.is_activated:
                    continue

                # If it has low confidence score
                if track.obs_history[-1][2] <= args.det_high_thresh:
                    continue

                # Filter detection with small box size, Since gt does not include small boxes
                w, h = track.tlwh[2:]
                if h * w <= img_h * img_w * args.min_box_size:
                    continue

                # Filter detections around border, Since gt does not include boxes around border
                x1, y1, x2, y2 = track.x1y1x2y2
                if x1 <= 5 or y1 <= 5 or x2 >= img_w - 5 or y2 >= img_h - 5:
                    continue

                # Append
                online_tracks_filtered[cam].append(track)

            # Class agnostic NMS, Since gt does not include overlapped boxes
            if 2 <= len(online_tracks_filtered[cam]):
                online_tracks_filtered[cam] = class_agnostic_nms(online_tracks_filtered[cam])

        # 合并轨迹
        online_tracks = []
        for cam in cams:
            online_tracks += online_tracks_filtered[cam]

        # 收集当前轨迹的全局ID
        online_global_ids = {'drone1': [], 'drone2': []}
        for track in online_tracks:
            if track.global_id is not None:
                online_global_ids[track.cam].append(track.global_id)


        # 获取外观特征并计算配对距离
        online_feats = np.array([track.get_feature(mode=args.get_feat_mode) for track in online_tracks])

        # 确保相机中有多个目标，否则不进行聚类
        if online_feats.ndim == 2:
            p_dists = pdist(online_feats, metric='cosine')
            p_dists = np.clip(p_dists, 0, 1)

            # 约束条件
            for i in range(len(online_tracks)):
                for j in range(i + 1, len(online_tracks)):
                    # 转换索引
                    idx = len(online_tracks) * i + j - ((i + 2) * (i + 1)) // 2

                    # 相同相机的轨迹
                    if online_tracks[i].cam == online_tracks[j].cam:
                        p_dists[idx] = 10
                        continue

                    # 如果目标不在重叠区域内 (i -> j)
                    overlap_region = overlap_regions[online_tracks[i].cam][online_tracks[j].cam]
                    x1, y1, x2, y2 = online_tracks[i].x1y1x2y2.astype(np.int32)
                    if overlap_region[y2, (x1 + x2) // 2] == 0:
                        p_dists[idx] = 10
                        continue

                    # 如果目标不在重叠区域内 (j -> i)
                    overlap_region = overlap_regions[online_tracks[j].cam][online_tracks[i].cam]
                    x1, y1, x2, y2 = online_tracks[j].x1y1x2y2.astype(np.int32)
                    if overlap_region[y2, (x1 + x2) // 2] == 0:
                        p_dists[idx] = 10
                        continue


            # 聚类 =================================================================================================
            # 生成聚类链接矩阵
            if np.all(p_dists == 0) == False:
                linkage_matrix = linkage(p_dists, method='complete')
                ranked_dists = np.sort(list(set(list(linkage_matrix[:, 2]))), axis=None)

                # 通过调整距离阈值并计算dunn索引生成聚类集合
                clusters, dunn_indices, c_dists = [], [], squareform(p_dists)
                for rdx in range(2, ranked_dists.shape[0] + 1):
                    if ranked_dists[-rdx] <= args.mtmc_match_thr:
                        clusters.append(fcluster(linkage_matrix, ranked_dists[-rdx] + 1e-5, criterion='distance') - 1)
                        dunn_indices.append(dunn(clusters[-1], c_dists))

                if len(clusters) == 0:
                    cluster = fcluster(linkage_matrix, ranked_dists[0] - 1e-5, criterion='distance') - 1
                else:
                    # 选择除不匹配对外连接最紧密的聚类集合
                    # 获取dunn索引的跳变点
                    dunn_indices.insert(0, 0)
                    pos = np.argmax(np.diff(dunn_indices))
                    cluster = clusters[pos]

                # 运行多相机追踪=====================================================================
                # 初始化
                num_cluster = len(list(set(list(cluster))))

                # 根据相同聚类集合中的其他轨迹分配全局ID给新轨迹
                for cdx in range(num_cluster):
                    track_idx = np.where(cluster == cdx)[0]

                    # 检查相同聚类集合中的索引和全局ID
                    infos = []
                    for tdx in track_idx:
                        if online_tracks[tdx].global_id is not None:
                            infos.append([tdx, online_tracks[tdx].global_id])

                    # 如果聚类集合中的某些轨迹已经有全局ID则分配相同的全局ID给新的轨迹
                    if len(infos) > 0:
                        # 分配全局ID，收集轨迹，更新外观特征
                        for tdx in track_idx:
                            if online_tracks[tdx].global_id is None:
                                # 根据节点间的最小距离获取全局ID并排序
                                sorted_infos = sorted(copy.deepcopy(infos), key=lambda x: c_dists[tdx, x[0]])
                                for info in sorted_infos:
                                    if online_tracks[info[0]].global_id not in online_global_ids[
                                        online_tracks[tdx].cam]:
                                        # 分配全局ID并收集轨迹
                                        online_tracks[tdx].global_id = info[1]
                                        clusters_dict[info[1]].add_track(online_tracks[tdx])
                                        break

                # 第二次聚类，将没有分配global_id的track之前的聚类集合clusters_dict再匹配
                # 获取当前剩余轨迹
                remain_tracks = [track for track in online_tracks if track.global_id is None]

                # 计算之前聚类集合与当前聚类集合之间的配对相似度距离
                dists = pairwise_tracks_dist(clusters_dict, remain_tracks, fdx, metric='cosine')

                # 运行匈牙利算法
                indices = linear_assignment(dists)

                # 根据阈值进行匹配
                for row, col in indices:
                    if dists[row, col] <= args.mtmc_match_thr \
                            and list(clusters_dict.keys())[row] not in online_global_ids[remain_tracks[col].cam]:
                        # 分配全局ID并收集轨迹
                        remain_tracks[col].global_id = list(clusters_dict.keys())[row]
                        clusters_dict[list(clusters_dict.keys())[row]].add_track(remain_tracks[col])

                # 如果没有匹配则分配新的ID
                for remain_track in remain_tracks:
                    if remain_track.global_id is None:
                        # 分配全局ID，收集轨迹
                        remain_track.global_id = next_global_id
                        clusters_dict[next_global_id] = Cluster()
                        clusters_dict[next_global_id].add_track(remain_track)

                        # ID自增
                        next_global_id += 1

                # 删除超过生存周期的ID
                del_key = [key for key in clusters_dict.keys() if
                           fdx - clusters_dict[key].end_frame > args.max_time_differ]
                for key in del_key:
                    del clusters_dict[key]

                # 输出追踪结果到result文件
                for track in online_tracks:
                    left, top, w, h = track.tlwh

                    # 扩展边框大小
                    cx, cy = left + w / 2, top + h / 2
                    w, h = w * 1.5, h * 1.5
                    left, top = cx - w / 2, cy - h / 2
                    # 过滤过小和过大的框
                    if w * h / img_w / img_h < 0.003 or 0.3 < w * h / img_w / img_h:
                        continue

                # 显示追踪结果
                if is_show_tracks:
                    for id in clusters_dict:
                        for j in range(len(clusters_dict[id].tracks)):
                            global_id = clusters_dict[id].tracks[j].global_id
                            cam = clusters_dict[id].tracks[j].cam
                            if cam == 'drone1':
                                draw_boxes(drone1_img, clusters_dict[id].tracks[j].tlwh, global_id)
                            if cam == 'drone2':
                                draw_boxes(drone2_img, clusters_dict[id].tracks[j].tlwh, global_id)

        end_time = time.time()
        track_time = end_time - start_time
        if track_time != 0:
            track_fps = 1 / (end_time - start_time)
            print("追踪帧率", track_fps)

        return drone1_img, drone2_img, label, detect1_info, detect2_info, track1_info, track2_info, detect_fps, track_fps








