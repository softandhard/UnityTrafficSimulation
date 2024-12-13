import os
import cv2
import time
import copy
import torch
import random
import numpy as np
from opts_visdrone import opt
from torchvision import transforms
from train_feat_ext.utils import dunn
from tracking.bot_sort import BoTSORT
from train_feat_ext.utils import LoadImages
from models.experimental import attempt_load
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from models.feature_extractor import FeatureExtractor
from train_feat_ext.utils import linear_assignment
from train_feat_ext.utils import check_img_size, non_max_suppression, scale_coords
from train_feat_ext.utils import letterbox, class_agnostic_nms, pairwise_tracks_dist

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

    @ property
    def end_frame(self):
        return np.max([track.end_frame for track in self.tracks])

    @ property
    def cam_list(self):
        return [track.cam for track in self.tracks]


def run_mtmc():
    # Load models ====================================================================================================
    # Load detection model
    det_model = attempt_load(opt.det_weights + opt.det_name + '.pt')
    det_model = det_model.cuda().eval().half()

    # For detection model
    stride = int(det_model.stride.max())
    img_size = opt.img_size.copy()
    img_size[0] = check_img_size(opt.img_size[0], s=stride)
    img_size[1] = check_img_size(opt.img_size[1], s=stride)

    # Load feature extraction model
    feat_ext_model = FeatureExtractor(opt.feat_ext_name, opt.avg_type, opt.feat_ext_weights)
    feat_ext_model = feat_ext_model.cuda().eval().half()

    # For feature extraction model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Prepare ========================================================================================================
    # Prepare output folder
    output_dir = opt.output_dir + '%s/' % opt.det_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare result txt file
    result_txt = open(output_dir + 'mtmc_visdrone_%s_%s.txt' % (opt.feat_ext_name, opt.avg_type), 'w')

    # Prepare others
    cams = os.listdir(opt.data_dir)
    datasets, trackers, f_nums = {}, {}, []

    print("cams==", cams)
    for cam in cams:
        # Prepare 1
        img_dir = os.path.join(opt.data_dir, cam) + '/frame/*'
        print("cam==", cam)
        datasets[cam] = iter(LoadImages(img_dir, img_size=img_size, stride=stride))
        trackers[cam] = BoTSORT(opt)
        f_nums.append(datasets[cam].nf)


    # Warm-up models
    with torch.autocast('cuda'):
        for _ in range(10):
            det_model(torch.rand((4, 3, img_size[0], img_size[1]), device='cuda').half())
            feat_ext_model(torch.rand((10, 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half())

    # Temporal alignment
    temp_align = {}
    for cam in cams:
        temp_align[cam] = {}
        for i in range(0, np.max(f_nums) + 1):
            # Default
            temp_align[cam][i] = 0

            # Set for each camera
            if cam == '1':
                temp_align[cam][i] = i + 1

            elif cam == '2':
                temp_align[cam][i] = i + 1



    # For time measurement
    total_times = {'Det': 0, 'Ext': 0, 'MTSC': 0, 'MTMC': 0}

    # Run ==============================================================================================================
    # Initialize
    img_h, img_w = opt.img_ori_size
    next_global_id, dunn_index_prev = 0, -1e5
    clusters_dict = {}

    # Run
    for fdx in range(0, np.max(f_nums) + 1):
        # Generate empty batches
        batch_img = torch.zeros((len(cams), 3, img_size[0], img_size[1]), device='cuda').half()
        batch_img_ori = torch.zeros((len(cams), 3, opt.img_ori_size[0], opt.img_ori_size[1]), device='cuda').half()
        batch_patch = torch.zeros((100 * len(cams), 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half()
        batch_img_cam = {}
        # Prepare images
        valid_cam = {}
        for cdx, cam in enumerate(cams):
            # Read
            valid_cam[cam] = True
            path, img, img_ori, _ = datasets[cam].__next__(cam, temp_align[cam][fdx])
            print("path==", path)
            batch_img_cam[cam] = img_ori
            # Check
            if img is None:
                valid_cam[cam] = False
                continue

            # Store
            batch_img[cdx] = torch.tensor(img / 255.0, device='cuda').half()
            batch_img_ori[cdx] = torch.tensor(img_ori.transpose((2, 0, 1)) / 255.0, device='cuda').half()

        start = time.time()

        # Detect =====================================================================================================
        with torch.autocast('cuda'):
            # batch_img 是输入的图像数据
            # opt.augment 是指定是否对输入图像进行数据增强
            # preds 是推理的结果，通常包括了检测到的目标的类别、位置和置信度等信息
            # 通过使用 [0]，代码选择了输出列表的第一个元素，可能是最相关或最高置信度的目标信息
            preds = det_model(batch_img[list(valid_cam.values())], augment=opt.augment)[0]

        # NMS
        preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres,
                                    classes=opt.classes, agnostic=opt.agnostic_nms)

        # Insert empty results
        for cdx, cam in enumerate(cams):
            if not valid_cam[cam]:
                preds.insert(cdx, torch.zeros((0, 6)).cuda().half())

        total_times['Det'] += time.time() - start
        start = time.time()

        # Prepare feature extraction =================================================================================
        # Prepare patches for feature extraction model
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

                    # Filter detection with box size
                    if (x2 - x1) * (y2 - y1) <= img_h * img_w * opt.min_box_size / 2:
                        continue

                    # Add detections
                    new_box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1), conf.item()])
                    new_box = new_box[np.newaxis, :]
                    detection[cams[pdx]] = np.concatenate([detection[cams[pdx]], new_box], axis=0)

                    # Get patch
                    patch = batch_img_ori[pdx][:, max(y1, 0):min(y2 + 1, img_h), max(x1, 0):min(x2 + 1, img_w)]
                    patch = normalize(letterbox(patch))
                    batch_patch[det_count] = patch
                    det_count += 1

        # Extract features
        with torch.autocast('cuda'):
            batch_patch = batch_patch[:det_count]
            batch_feat = feat_ext_model(batch_patch)
        batch_feat = batch_feat.squeeze().cpu().numpy()

        total_times['Ext'] += time.time() - start
        start = time.time()

        # Multi-target Single-Camera Tracking ========================================================================
        # Separate features
        feat_count, feat = 0, {}
        for cam in cams:
            feat[cam] = batch_feat[feat_count:feat_count + len(detection[cam])]
            feat_count += len(detection[cam])

        # Run Multi-target Single-Camera Tracking and online tracks
        online_tracks_raw = {}
        for cam in cams:
            online_tracks_raw[cam] = trackers[cam].update(cam, detection[cam], feat[cam])

        total_times['MTSC'] += time.time() - start
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
        online_global_ids = {'1': [], '2': []}
        for track in online_tracks:
            if track.global_id is not None:
                online_global_ids[track.cam].append(track.global_id)

        # Get features and calculate pairwise distances
        online_feats = np.array([track.get_feature(mode=opt.get_feat_mode) for track in online_tracks])
        p_dists = pdist(online_feats, metric='cosine')
        p_dists = np.clip(p_dists, 0, 1)

        # Apply constraints
        for i in range(len(online_tracks)):
            for j in range(i + 1, len(online_tracks)):
                # Covert index
                idx = len(online_tracks) * i + j - ((i + 2) * (i + 1)) // 2

                # If same camera
                if online_tracks[i].cam == online_tracks[j].cam:
                    p_dists[idx] = 10
                    continue


        # Clustering =================================================================================================
        # Generate linkage matrix with hierarchical clustering
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

        # Run Multi-target Multi-Camera Tracking =====================================================================
        # Initialize
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
                            if online_tracks[info[0]].global_id not in online_global_ids[online_tracks[tdx].cam]:
                                # Assign global id, Collect
                                online_tracks[tdx].global_id = info[1]
                                clusters_dict[info[1]].add_track(online_tracks[tdx])
                                break

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
        del_key = [key for key in clusters_dict.keys() if fdx - clusters_dict[key].end_frame > opt.max_time_differ]
        for key in del_key:
            del clusters_dict[key]

        total_times['MTMC'] += time.time() - start

        for id in clusters_dict:
            cam_list = clusters_dict[id].cam_list
            print("cam-list==", cam_list)
            # tracks是全局ID id中包含的轨迹集合，tracks== [OT_1_(0-1)]
            print("tracks==", clusters_dict[id].tracks)
            if not os.path.exists("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/1"):
                os.makedirs("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/1")
            if not os.path.exists("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/2"):
                os.makedirs("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/2")

            for j in range(len(clusters_dict[id].tracks)):
                global_id = clusters_dict[id].tracks[j].global_id
                cam = clusters_dict[id].tracks[j].cam
                print("global-id", global_id)
                print("track-cam", cam)
                # ltwh：1005 367 294 178
                draw_img = draw_boxes(batch_img_cam[cam], clusters_dict[id].tracks[j].tlwh, global_id)
                cv2.imwrite("C:/Users/cjh/Desktop/MTMCT/online_MTMC/outputs/img_result/{}/{}.jpg".format(cam, global_id), draw_img)

        # Logging
        for track in online_tracks:
            left, top, w, h = track.tlwh

            # Expand box, Since gt boxes are not tightly annotated around objects and quite larger than objects
            cx, cy = left + w / 2, top + h / 2
            w, h = w * 1.5, h * 1.5
            left, top = cx - w / 2, cy - h / 2

            # Filter with size, Since gt does not include small boxes
            if w * h / img_w / img_h < 0.003 or 0.3 < w * h / img_w / img_h:
                continue

            print('%d %d %d %d %d %d %d -1 -1' % (int(track.cam[-1]), track.global_id, temp_align[track.cam][fdx],
                                                  int(left), int(top), int(w), int(h)), file=result_txt)

    # Logging
    track_t, total_t = 0, 0
    print('%s_%s_%s' % (opt.det_name, opt.feat_ext_name, opt.avg_type))
    for key in total_times.keys():
        print('%s: %05f' % (key, total_times[key] / (np.max(f_nums) + 1)))
        track_t += total_times[key] / (np.max(f_nums) + 1) if key == 'MTSC' or key == 'MTMC' else 0
        total_t += total_times[key] / (np.max(f_nums) + 1)
    print('Tracking Time: %05f' % track_t)
    print('Total Time: %05f' % total_t)


if __name__ == '__main__':
    with torch.no_grad():
        run_mtmc()
