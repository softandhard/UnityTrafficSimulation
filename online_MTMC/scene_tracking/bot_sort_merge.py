import numpy as np
from online_MTMC.scene_tracking import matching
from online_MTMC.scene_tracking.kalman_filter import KalmanFilter
from online_MTMC.scene_tracking.fixed_kalman_filter import FixedKalmanFilter
from online_MTMC.scene_tracking.track_merge import TrackState, BaseTrack, Track
from online_MTMC.opts_merge import opt

def joint_tracks(t_list_a, t_list_b):
    exists = {}
    res = []

    for t in t_list_a:
        exists[t.track_id] = 1
        res.append(t)

    for t in t_list_b:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)

    return res


def sub_tracks(t_list_a, t_list_b):
    res = {}

    for t in t_list_a:
        res[t.track_id] = t

    for t in t_list_b:
        tid = t.track_id
        if res.get(tid, 0):
            del res[tid]

    return list(res.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    pdist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(pdist < 0.15)
    dup_a, dup_b = list(), list()

    for p, q in zip(*pairs):
        time_p = tracks_a[p].frame_id - tracks_a[p].start_frame
        time_q = tracks_b[q].frame_id - tracks_b[q].start_frame

        if time_p > time_q:
            dup_b.append(q)
        else:
            dup_a.append(p)

    res_a = [t for i, t in enumerate(tracks_a) if not i in dup_a]
    res_b = [t for i, t in enumerate(tracks_b) if not i in dup_b]

    return res_a, res_b


class BoTSORT(object):
    def __init__(self, opt):
        # Initialize
        self.tracked = []
        self.lost = []
        self.finished = []
        self.kalman_filter = KalmanFilter()
        self.unsecented_kalman_filter = FixedKalmanFilter()

        BaseTrack.clear_count()

        # Set parameters
        self.opt = opt
        self.frame_id = -1
        # 轨迹保存的最大时间，等同于deepsort中的max_age
        self.max_time_lost = int(opt.max_time_lost)

    def update(self, cam, detections, features, scene_features):
        # Initialize
        activated = []
        re_activated = []
        lost = []
        finished = []
        self.frame_id += 1

        if len(detections.shape) == 1:
            detections = detections[np.newaxis, :]

        # Initialize
        boxes = detections[:, 0:4].astype(np.float32)
        confidences = detections[:, 4].astype(np.float32)
        features = features.astype(np.float32)
        scene_features = scene_features.astype(np.float32)

        # Remove bad detections
        # 移除置信度过低的检测框，置信度小于self.opt.det_low_thresh
        # indices_remain为一个布尔型数组，该数组中的元素为True表示对应位置的置信度大于self.opt.det_low_thresh，
        # 为False表示置信度小于等于self.opt.det_low_thresh
        # 保留满足条件的元素，并更新boxes、confidences和features，使其仅包含置信度高于阈值的检测结果
        indices_remain = confidences > self.opt.det_low_thresh
        boxes = boxes[indices_remain]
        confidences = confidences[indices_remain]
        features = features[indices_remain]

        scene_features = scene_features[indices_remain]

        # Find high confidence detections
        # 获取高置信度的检测框，置信度大于 self.opt.det_high_thresh
        # indices_high为一个布尔型数组
        indices_high = confidences > self.opt.det_high_thresh
        boxes_first = boxes[indices_high]
        confidences_first = confidences[indices_high]
        features_first = features[indices_high]

        scene_features_first = scene_features[indices_high]

        # Encode detections with Track
        # 将检测框 boxes_first 以Track格式表示为 detections_first
        # 原来的 botsort 中采用self.args.with_reid控制是否使用特征提取模型
        # 使用reid模型的detections会包含目标外观特征信息features_keep
        # if len(dets) > 0:
        #     '''Detections'''
        #     if self.args.with_reid:
        #         detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
        #                       (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
        #     else:
        #         detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
        #                       (tlbr, s) in zip(dets, scores_keep)]
        if len(boxes_first) > 0:
            detections_first = [Track(cam, cxcywh, s, f, scene_f)
                                for (cxcywh, s, f, scene_f) in
                                zip(boxes_first, confidences_first, features_first, scene_features_first)]
        else:
            detections_first = []

        # Split into unactivated (tracked only 1 beginning frame) and tracked (tracked more than 2 frames)
        tracked, unactivated = [], []
        for track in self.tracked:
            if not track.is_activated:
                unactivated.append(track)
            else:
                tracked.append(track)

        # Step 1 - First association, tracks & high confidence detection boxes =========================================
        # Merge tracked (tracked more than 2 frames) and lost (tracked more than 2 frames, lost tracks)
        pool = joint_tracks(tracked, self.lost)

        # 原来的botsort中包含了相机运动校正处理
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)
        # 对图像进行几何校正，然后将校正后的图像应用到目标轨迹上，以修复相机运动引起的影响
        # Predict the current location with KF
        Track.multi_predict(pool)

        # Calculate cosine and IoU distances
        # cos_dists == ndarray:(n ,m) n:tracked 之前检测到的轨迹特征, m:detections 现在检测到的轨迹特征
        # 距离矩阵中的值越小，表示两个数据样本之间的距离越近，而值越大表示两个数据样本之间的距离越远
        # 对于欧氏距离、曼哈顿距离等，数值越小表示距离越近；对于余弦距离，数值越接近于0表示距离越近
        # iou_dists值越接近0，即重叠更多，匹配效果更好
        cos_dists, scene_cos_dists = matching.embedding_distance(pool, detections_first)
        iou_dists = matching.iou_distance(pool, detections_first)

        # Distance
        # dists，scene_dists中的数值越小的表示两个轨迹之间的距离越近，反之则表示距离越远
        dists = cos_dists.copy()
        scene_dists = scene_cos_dists.copy()
        dists_merge = cos_dists.copy()

        if opt.merge_pattern == "minimal":
            # 最小值
            for i in range(dists.shape[0]):
                for j in range(dists.shape[1]):
                    if abs(dists[i, j]) > abs(scene_dists[i, j]):
                        dists_merge[i, j] = scene_dists[i, j]
                    else:
                        dists_merge[i, j] = dists[i, j]

        if opt.merge_pattern == "average":
            # 平均值
            for i in range(dists.shape[0]):
                for j in range(dists.shape[1]):
                    dists_merge[i, j] = (scene_dists[i, j] + dists[i, j]) / 2


        # 加权平均值
        # for i in range(scene_dists.shape[0]):
        #     for j in range(scene_dists.shape[1]):
        #         dists_merge[i, j] = 0.8 * scene_dists[i, j] + 0.2 * dists[i, j]

        dists_merge[dists_merge > self.opt.cos_thr] = 1.
        dists_merge[iou_dists > 1 - (1 - self.opt.iou_thr) / 2] = 1.

        # Associate
        matches, u_track, u_detection = matching.linear_assignment(dists_merge, thresh=self.opt.cos_thr)

        # Update state
        for t, d in matches:
            track = pool[t]
            det = detections_first[d]

            if track.state == TrackState.Tracked:
                track.update(detections_first[d], self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                re_activated.append(track)

        # Step 2 - Second association, left tracks & low confidence detection boxes ====================================
        # Find low confidence detections
        # 获取低置信度的检测框，置信度在范围（self.opt.det_low_thresh， self.opt.det_high_thresh）内
        indices_high = confidences < self.opt.det_high_thresh
        indices_low = confidences > self.opt.det_low_thresh
        indices_second = np.logical_and(indices_low, indices_high)
        boxes_second = boxes[indices_second]
        confidences_second = confidences[indices_second]
        features_second = features[indices_second]

        scene_features_second = scene_features[indices_second]

        # Encode detections with Track
        if len(boxes_second) > 0:
            detections_second = [Track(cam, x1y1x2y2, s, f, scene_f)
                                 for (x1y1x2y2, s, f, scene_f) in
                                 zip(boxes_second, confidences_second, features_second, scene_features_second)]
        else:
            detections_second = []

        # Calculate distances
        # iou_dists值越接近0，即重叠更多，匹配效果更好
        remained = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        iou_dists = matching.iou_distance(remained, detections_second)

        # Calculate cosine and IoU distances
        dists = iou_dists.copy()
        dists[iou_dists > self.opt.iou_thr] = 1.

        # Associate
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.opt.iou_thr)

        # Update state
        for t, d in matches:
            track = remained[t]
            det = detections_second[d]

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                re_activated.append(track)

        # Find lost tracks
        for t in u_track:
            track = remained[t]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # Step 3 - Third association, unactivated tracks & left high confidence detection boxes ========================
        # Get left high confidence detections
        detections_first = [detections_first[i] for i in u_detection]

        # Calculate distances
        # iou_dists值越接近0，即重叠更多，匹配效果更好
        cos_dists, scene_cos_dists = matching.embedding_distance(unactivated, detections_first)
        iou_dists = matching.iou_distance(unactivated, detections_first)

        # Distance
        dists = cos_dists.copy()
        scene_dists = scene_cos_dists.copy()
        dists_merge = cos_dists.copy()

        if opt.merge_pattern == "minimal":
            # 最小值
            for i in range(dists.shape[0]):
                for j in range(dists.shape[1]):
                    if abs(dists[i, j]) > abs(scene_dists[i, j]):
                        dists_merge[i, j] = scene_dists[i, j]
                    else:
                        dists_merge[i, j] = dists[i, j]


        if opt.merge_pattern == "average":
            # 平均值
            for i in range(dists.shape[0]):
                for j in range(dists.shape[1]):
                    dists_merge[i, j] = (scene_dists[i, j] + dists[i, j]) / 2

        # 加权平均值
        # for i in range(dists.shape[0]):
        #     for j in range(dists.shape[1]):
        #         dists_merge[i, j] = 0.8 * scene_dists[i, j] + 0.2 * dists[i, j]

        dists_merge[dists_merge > self.opt.cos_thr] = 1.
        dists_merge[iou_dists > 1 - (1 - self.opt.iou_thr) / 2] = 1.

        # dists[cos_dists > self.opt.cos_thr] = 1.
        # dists[iou_dists > 1 - (1 - self.opt.iou_thr) / 2] = 1.

        # Associate
        matches, u_unactivated, u_detection = matching.linear_assignment(dists_merge, thresh=self.opt.cos_thr)

        # Update state
        for t, d in matches:
            unactivated[t].update(detections_first[d], self.frame_id)
            activated.append(unactivated[t])

        # Update state
        for it in u_unactivated:
            track = unactivated[it]
            track.mark_removed()

        # Initiate new tracks
        for n in u_detection:
            track = detections_first[n]
            if track.confidence >= self.opt.det_high_thresh:
                # Exclude detection with small box size, Since gt does not include small boxes
                w, h = track.tlwh[2:]
                img_h, img_w = self.opt.img_ori_size
                if h * w <= img_h * img_w * self.opt.min_box_size:
                    continue

                # Initiate new track
                track.initiate(self.kalman_filter, self.unsecented_kalman_filter, self.frame_id)
                activated.append(track)

        # Update state
        for track in self.lost:
            # Finish track temporal distance
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_finished()
                finished.append(track)

            # Finish lost track with small box size, Since gt does not include small boxes
            w, h = track.tlwh[2:]
            img_h, img_w = self.opt.img_ori_size
            if h * w <= img_h * img_w * self.opt.min_box_size:
                track.mark_finished()
                finished.append(track)

        # Merge 1
        # 将已经确认的轨迹加入self.tracked
        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        # 将已经匹配的和重新匹配上的轨迹加入self.tracked
        self.tracked = joint_tracks(self.tracked, activated)
        self.tracked = joint_tracks(self.tracked, re_activated)

        # Merge 2
        self.lost.extend(lost)
        self.finished.extend(finished)

        # Merge 3
        self.lost = sub_tracks(self.lost, self.tracked)
        self.lost = sub_tracks(self.lost, self.finished)
        self.tracked, self.lost = remove_duplicate_tracks(self.tracked, self.lost)

        # 返回确认追踪到的轨迹
        # 包含检测器检测到的目标信息，相机号，全局ID。目标外观特征，卡尔曼滤波器，均值，方差
        # self.cam = cam
        # self.global_id = None
        # self.cxcywh = cxcywh
        # self.confidence = confidence
        # self.curr_feat = feat
        # self.smooth_feat = None
        # self.kalman_filter = None
        # self.mean, self.covariance = None, None
        return self.tracked
