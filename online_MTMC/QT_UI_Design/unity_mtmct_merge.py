import os
import socket
import cv2
import time
import copy
import torch
import random
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
def receive_unity_data(client_socket, show_vid):
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

    if show_vid:
        cv2.imshow("Received Image", image)
        cv2.imshow("Received Image--2", image2)
        cv2.imwrite("C:/Users/cjh/Desktop/MTMCT/online_MTMC/QT_UI_Design/outputs/res.png", image)
        cv2.imwrite("C:/Users/cjh/Desktop/MTMCT/online_MTMC/QT_UI_Design/outputs/res2.png", image2)

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


def run_mtmc():
    # 加载模型 ====================================================================================================
    # 加载检测模型
    det_model = attempt_load(opt.det_weights + opt.det_name + '.pt')
    det_model = det_model.cuda().eval().half()

    # 设置检测模型参数
    stride = int(det_model.stride.max())
    img_size = opt.img_size.copy()
    img_size[0] = check_img_size(opt.img_size[0], s=stride)
    img_size[1] = check_img_size(opt.img_size[1], s=stride)


    # 加载特征提取模型
    feat_ext_model = FeatureExtractor(opt.feat_ext_name, opt.avg_type, opt.feat_ext_weights)
    feat_ext_model = feat_ext_model.cuda().eval().half()

    # 加载特征提取模型
    scene_feat_ext_model = FeatureExtractor_Deepsort(opt.scene_feat_ext_name, opt.scene_avg_type, opt.scene_feat_ext_weights)
    scene_feat_ext_model = scene_feat_ext_model.cuda().eval().half()
    # 归一化处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 准备环境 ========================================================================================================
    # 其他准备处理
    datasets, trackers, f_nums = {}, {}, []
    roi_masks, overlap_regions = {}, {}

    cams = ['drone1', 'drone2']
    for cam in cams:
        roi_masks[cam] = cv2.imread('C:/Users/cjh/Desktop/MTMCT/online_MTMC/QT_UI_Design/rois/%s.jpg' % cam, cv2.IMREAD_GRAYSCALE)
        overlap_regions[cam] = {}
        for cam_ in cams:
            overlap_regions[cam][cam_] = cv2.imread('C:/Users/cjh/Desktop/MTMCT/online_MTMC/QT_UI_Design/overlap_zones/%s_%s.jpg' % (cam, cam_),
                                                    cv2.IMREAD_GRAYSCALE) if cam_ != cam else None

        # 准备追踪器
        print("cam==", cam)
        trackers[cam] = BoTSORT(opt)

    # 预热模型
    with torch.autocast('cuda'):
        for _ in range(10):
            det_model(torch.rand((4, 3, img_size[0], img_size[1]), device='cuda').half())
            feat_ext_model(torch.rand((10, 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half())
            scene_feat_ext_model(torch.rand((10, 3, opt.part_patch_size[0], opt.part_patch_size[1]), device='cuda').half())

    client_socket = create_unity_connection()
    drone1_img, drone2_img = receive_unity_data(client_socket, opt.show_img)

    # 运行时间测量
    total_times = {'Det': 0, 'Ext': 0, 'MTSC': 0, 'MTMC': 0}

    # 运行配置 ==============================================================================================================
    # 初始化
    img_h, img_w = opt.img_ori_size
    next_global_id, dunn_index_prev = 0, -1e5
    clusters_dict = {}
    fdx = 0
    while drone1_img is not None and drone2_img is not None:
        # Run
        # 生成空batch
        batch_img = torch.zeros((len(cams), 3, img_size[0], img_size[1]), device='cuda').half()
        batch_img_ori = torch.zeros((len(cams), 3, opt.img_ori_size[0], opt.img_ori_size[1]), device='cuda').half()
        batch_patch = torch.zeros((100 * len(cams), 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half()
        batch_part_patch = torch.zeros((100 * len(cams), 3, opt.part_patch_size[0], opt.part_patch_size[1]), device='cuda').half()
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

        start = time.time()

        # 检测图像 =====================================================================================================
        with torch.autocast('cuda'):
            # batch_img 是输入的图像数据
            # opt.augment 是指定是否对输入图像进行数据增强
            # preds 是推理的结果，通常包括了检测到的目标的类别、位置和置信度等信息
            # 通过使用 [0]，代码选择了输出列表的第一个元素，可能是最相关或最高置信度的目标信息
            preds = det_model(batch_img[list(valid_cam.values())], augment=opt.augment)[0]

        # 非极大值抑制
        preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres,
                                    classes=opt.classes, agnostic=opt.agnostic_nms)

        # 如果图像为空，插入空张量
        for cdx, cam in enumerate(cams):
            if not valid_cam[cam]:
                preds.insert(cdx, torch.zeros((0, 6)).cuda().half())

        total_times['Det'] = time.time() - start
        start = time.time()

        # 重新获取unity图像
        drone1_img, drone2_img = receive_unity_data(client_socket, opt.show_img)

        # 准备提取特征 =================================================================================
        # 准备特征提取模型
        det_count, detection = 0, {}
        for pdx, pred in enumerate(preds):
            # Prepare dictionary to store detection results
            detection[cams[pdx]] = np.zeros((0, 5))

            # If there are valid predictions
            if len(pred) > 0:
                # Rescale boxes from img_size to im0s size
                pred[:, :4] = scale_coords(batch_img.shape[2:], pred[:, :4], batch_img_ori.shape[2:4])

                # Post-process detections
                for *xyxy, conf, _ in reversed(pred):
                    # Convert to integer
                    x1, y1 = round(xyxy[0].item()), round(xyxy[1].item())
                    x2, y2 = round(xyxy[2].item()), round(xyxy[3].item())

                    # Filter detections with RoI mask
                    if roi_masks[cams[pdx]][min(y2 + 1, img_h) - 1, (max(x1, 0) + min(x2 + 1, img_w)) // 2] == 0:
                        continue

                    # Filter detection with box size
                    if (x2 - x1) * (y2 - y1) <= img_h * img_w * opt.min_box_size / 2:
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

        # 提取特征
        with torch.autocast('cuda'):
            batch_patch = batch_patch[:det_count]
            batch_feat = feat_ext_model(batch_patch)
        batch_feat = batch_feat.squeeze().cpu().numpy()
        # 提取特征
        with torch.autocast('cuda'):
            batch_part_patch = batch_part_patch[:det_count]
            batch_scene_feat = scene_feat_ext_model(batch_part_patch)
        batch_scene_feat = batch_scene_feat.squeeze().cpu().numpy()

        total_times['Ext'] = time.time() - start
        start = time.time()

        # 单相机追踪 ========================================================================
        # 分离特征
        feat_count, feat = 0, {}
        scene_count, scene_feat = 0, {}
        for cam in cams:
            feat[cam] = batch_feat[feat_count:feat_count + len(detection[cam])]
            feat_count += len(detection[cam])

            scene_feat[cam] = batch_scene_feat[scene_count:scene_count + len(detection[cam])]
            scene_count += len(detection[cam])

        # Run Multi-target Single-Camera Tracking and online tracks
        online_tracks_raw = {}
        for cam in cams:
            online_tracks_raw[cam] = trackers[cam].update(cam, detection[cam], feat[cam], scene_feat[cam])

        total_times['MTSC'] = time.time() - start
        start = time.time()

        # Prepare Multi-target Multi-Camera Tracking =================================================================
        # Filter tracks
        online_tracks_filtered = {}
        for cam in cams:
            online_tracks_filtered[cam] = []
            for track in online_tracks_raw[cam]:
                # If not activated
                if not track.is_activated:
                    continue

                # If it has low confidence score
                if track.obs_history[-1][2] <= opt.det_high_thresh:
                    continue

                # Filter detection with small box size, Since gt does not include small boxes
                w, h = track.tlwh[2:]
                if h * w <= img_h * img_w * opt.min_box_size:
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

        # Merge
        online_tracks = []
        for cam in cams:
            online_tracks += online_tracks_filtered[cam]

        # Gather current tracking global ids
        online_global_ids = {'drone1': [], 'drone2': []}
        for track in online_tracks:
            if track.global_id is not None:
                online_global_ids[track.cam].append(track.global_id)


        # Get features and calculate pairwise distances
        online_feats = np.array([track.get_feature(mode=opt.get_feat_mode) for track in online_tracks])

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

                    # If the objects are not in overlapping region (i -> j)
                    overlap_region = overlap_regions[online_tracks[i].cam][online_tracks[j].cam]
                    x1, y1, x2, y2 = online_tracks[i].x1y1x2y2.astype(np.int32)
                    if overlap_region[y2, (x1 + x2) // 2] == 0:
                        p_dists[idx] = 10
                        continue

                    # If the objects are not in overlapping region (j -> i)
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

                # Observe clusters with adjusting a distance threshold and calculate dunn index
                clusters, dunn_indices, c_dists = [], [], squareform(p_dists)
                for rdx in range(2, ranked_dists.shape[0] + 1):
                    if ranked_dists[-rdx] <= opt.mtmc_match_thr:
                        clusters.append(fcluster(linkage_matrix, ranked_dists[-rdx] + 1e-5, criterion='distance') - 1)
                        dunn_indices.append(dunn(clusters[-1], c_dists))

                if len(clusters) == 0:
                    cluster = fcluster(linkage_matrix, ranked_dists[0] - 1e-5, criterion='distance') - 1
                else:
                    # Choose the most connected cluster except inappropriate pairs
                    # Get the index of the dunn indices where the values suddenly jump.
                    dunn_indices.insert(0, 0)
                    pos = np.argmax(np.diff(dunn_indices))
                    cluster = clusters[pos]

                # 运行多相机追踪=====================================================================
                # 初始化
                num_cluster = len(list(set(list(cluster))))

                # Assign global id to new tracks using other tracks in the same cluster
                for cdx in range(num_cluster):
                    track_idx = np.where(cluster == cdx)[0]

                    # Check index and global id of tracks in same cluster
                    infos = []
                    for tdx in track_idx:
                        if online_tracks[tdx].global_id is not None:
                            infos.append([tdx, online_tracks[tdx].global_id])

                    # If some tracks in the cluster already has global id, assign same global id to new tracks
                    if len(infos) > 0:
                        # Assign global id, Collect tracks, Update feature
                        for tdx in track_idx:
                            if online_tracks[tdx].global_id is None:
                                # Sort and get global id with the node with minimum distance
                                sorted_infos = sorted(copy.deepcopy(infos), key=lambda x: c_dists[tdx, x[0]])
                                for info in sorted_infos:
                                    if online_tracks[info[0]].global_id not in online_global_ids[
                                        online_tracks[tdx].cam]:
                                        # Assign global id, Collect
                                        online_tracks[tdx].global_id = info[1]
                                        clusters_dict[info[1]].add_track(online_tracks[tdx])
                                        break

                # 第二次聚类，将没有分配global_id的track之前的聚类集合clusters_dict再匹配
                # Get remaining current tracks
                remain_tracks = [track for track in online_tracks if track.global_id is None]

                # Calculate pairwise distance between previous clusters and current clusters
                dists = pairwise_tracks_dist(clusters_dict, remain_tracks, fdx, metric='cosine')

                # Run Hungarian algorithm
                indices = linear_assignment(dists)

                # Match with thresholding
                for row, col in indices:
                    if dists[row, col] <= opt.mtmc_match_thr \
                            and list(clusters_dict.keys())[row] not in online_global_ids[remain_tracks[col].cam]:
                        # Assign global id, Collect track
                        remain_tracks[col].global_id = list(clusters_dict.keys())[row]
                        clusters_dict[list(clusters_dict.keys())[row]].add_track(remain_tracks[col])

                # If not matched newly starts
                for remain_track in remain_tracks:
                    if remain_track.global_id is None:
                        # Assign global id, Collect track
                        remain_track.global_id = next_global_id
                        clusters_dict[next_global_id] = Cluster()
                        clusters_dict[next_global_id].add_track(remain_track)

                        # Increase
                        next_global_id += 1

                # Delete too old cluster
                del_key = [key for key in clusters_dict.keys() if
                           fdx - clusters_dict[key].end_frame > opt.max_time_differ]
                for key in del_key:
                    del clusters_dict[key]

                total_times['MTMC'] = time.time() - start

                # 输出追踪结果到result文件
                for track in online_tracks:
                    left, top, w, h = track.tlwh

                    # Expand box, Since gt boxes are not tightly annotated around objects and quite larger than objects
                    cx, cy = left + w / 2, top + h / 2
                    w, h = w * 1.5, h * 1.5
                    left, top = cx - w / 2, cy - h / 2
                    # Filter with size, Since gt does not include small boxes
                    if w * h / img_w / img_h < 0.003 or 0.3 < w * h / img_w / img_h:
                        continue

                for id in clusters_dict:
                    if not os.path.exists("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/drone1"):
                        os.makedirs("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/drone1")
                    if not os.path.exists("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/drone2"):
                        os.makedirs("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/drone2")

                    # tracks是全局ID id中包含的轨迹集合，tracks== [OT_1_(0-1)]
                    for j in range(len(clusters_dict[id].tracks)):
                        global_id = clusters_dict[id].tracks[j].global_id
                        cam = clusters_dict[id].tracks[j].cam

                        # ltwh：1005 367 294 178
                        draw_img = draw_boxes(batch_img_cam[cam], clusters_dict[id].tracks[j].tlwh, global_id)
                        cv2.imwrite("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/{}/{}.jpg".format(cam, global_id),
                            draw_img)

                # Logging
                track_t, total_t = 0, 0
                print('%s_%s_%s' % (opt.det_name, opt.feat_ext_name, opt.avg_type))
                for key in total_times.keys():
                    print('%s: %05f' % (key, total_times[key]))
                    track_t += total_times[key] if key == 'MTSC' or key == 'MTMC' else 0
                    total_t += total_times[key]
                print('Tracking Time: %05f' % track_t)
                print('4-Image Total Time: %05f' % total_t)
                print('single-Image Total Time: %05f' % (total_t / 4))



if __name__ == '__main__':
    with torch.no_grad():
        run_mtmc()
