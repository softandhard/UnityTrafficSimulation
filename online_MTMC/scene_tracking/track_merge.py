import copy
import numpy as np
from online_MTMC.scene_tracking.kalman_filter import KalmanFilter
from online_MTMC.scene_tracking.fixed_kalman_filter import FixedKalmanFilter

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Finished = 3
    Removed = 3


class BaseTrack(object):
    _count = 0
    track_id = 0
    frame_id = -1
    is_activated = False
    state = TrackState.New

    confidence = 0
    start_frame = 0
    time_since_update = 0

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_finished(self):
        self.state = TrackState.Finished

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0


class Track(BaseTrack):
    shared_kalman = KalmanFilter()
    unsecented_kalman = FixedKalmanFilter()

    def __init__(self, cam, cxcywh, confidence, feat=None, scene_feat=None):
        # Initialize basics
        self.cam = cam
        self.global_id = None
        self.cxcywh = cxcywh
        self.confidence = confidence
        self.future_len = 5
        self.img_size = (1080, 1920)

        # Initialize about feature
        # τhigh, τ1, τ2, α, and Tmax are empirically set to
        # 0.6, 0.6, 0.8, 0.9, and 30
        self.alpha = 0.9
        self.curr_feat = feat
        self.curr_scene_feat = scene_feat
        self.smooth_feat = None
        self.smooth_scene_feat = None
        self.update_features(feat)
        self.update_scene_features(scene_feat)

        # Initialize others
        self.obs_history = []
        self.obs_scene_history = []
        self.is_activated = False
        self.kalman_filter = None
        self.unsecented_kalman_filter = None
        self.mean, self.covariance = None, None
        self.scene_mean, self.scene_covariance = None, None

    def update_features(self, feat):
        if feat is not None:
            # Normalize
            feat /= np.linalg.norm(feat)

            # Update
            if self.smooth_feat is None:
                self.smooth_feat = feat
            else:
                self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat

            # Normalize
            self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_scene_features(self, feat):
        if feat is not None:
            # Normalize
            feat /= np.linalg.norm(feat)

            # Update
            if self.smooth_scene_feat is None:
                self.smooth_scene_feat = feat
            else:
                self.smooth_scene_feat = self.alpha * self.smooth_scene_feat + (1 - self.alpha) * feat

            # Normalize
            self.smooth_scene_feat /= np.linalg.norm(self.smooth_scene_feat)

    def predict(self):
        mean_state = self.mean.copy()
        scene_mean_state = self.scene_mean.copy()

        # Give zero to vw and vh
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

            scene_mean_state[6] = 0
            scene_mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.scene_mean, self.scene_covariance = self.unsecented_kalman_filter.predict(scene_mean_state, self.scene_covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in tracks])
            multi_covariance = np.asarray([st.covariance for st in tracks])
            scene_multi_mean = np.asarray([st.scene_mean.copy() for st in tracks])
            scene_multi_covariance = np.asarray([st.scene_covariance.copy() for st in tracks])

            # Give zero to vw and vh
            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0

                    scene_multi_mean[i][6] = 0
                    scene_multi_mean[i][7] = 0

            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)
            scene_multi_mean, scene_multi_covariance = Track.unsecented_kalman.multi_predict(scene_multi_mean, scene_multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

            for i, (mean, cov) in enumerate(zip(scene_multi_mean, scene_multi_covariance)):
                tracks[i].scene_mean = mean
                tracks[i].scene_covariance = cov

    def initiate(self, kalman_filter, unsecented_kalman_filter, frame_id):
        # Start a new track
        self.track_id = self.next_id()
        self.kalman_filter = kalman_filter
        self.unsecented_kalman_filter = unsecented_kalman_filter

        # Update, Save
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh)
        self.scene_mean, self.scene_covariance = self.unsecented_kalman_filter.initiate(self.cxcywh)
        self.obs_history = [[frame_id, self.cxcywh.copy(), self.confidence, self.curr_feat.copy(),
                             self.mean.copy(), self.covariance.copy()]]

        self.obs_scene_history = [[frame_id, self.cxcywh.copy(), self.confidence, self.curr_scene_feat.copy(),
                             self.scene_mean.copy(), self.scene_covariance.copy()]]

        # Set
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = TrackState.Tracked
        self.is_activated = True if frame_id == 0 else self.is_activated


    def update(self, new_det, frame_id):
        self.frame_id = frame_id
        if self.state == TrackState.Tracked:
            new_det.confidence = np.clip(new_det.confidence + 0.01, 0, 1)
        if self.state == TrackState.Lost:
            new_det.confidence = np.clip(new_det.confidence - 0.01, 0, 1)
        if self.state == TrackState.Removed or self.state == TrackState.Finished:
            new_det.confidence = np.clip(new_det.confidence - 0.01, 0, 1)

        # Update
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               new_det.cxcywh, new_det.confidence)
        self.scene_mean, self.scene_covariance = self.unsecented_kalman_filter.update(self.mean, self.covariance, new_det.cxcywh)
        self.obs_history.append([frame_id, new_det.cxcywh.copy(), new_det.confidence, copy.deepcopy(new_det.curr_feat),
                                 self.mean.copy(), self.covariance.copy()])

        self.obs_scene_history.append([frame_id, new_det.cxcywh.copy(), new_det.confidence, copy.deepcopy(new_det.curr_scene_feat),
                                 self.scene_mean.copy(), self.scene_covariance.copy()])

        if new_det.curr_feat is not None:
            self.update_features(new_det.curr_feat)
        if new_det.curr_scene_feat is not None:
            self.update_scene_features(new_det.curr_scene_feat)

        # Set
        self.is_activated = True
        self.confidence = new_det.confidence
        self.state = TrackState.Tracked

    def re_activate(self, new_det, frame_id, new_id=False):
        if self.state == TrackState.Tracked:
            new_det.confidence = np.clip(new_det.confidence + 0.01, 0, 1)
        if self.state == TrackState.Lost:
            new_det.confidence = np.clip(new_det.confidence - 0.01, 0, 1)
        if self.state == TrackState.Removed or self.state == TrackState.Finished:
            new_det.confidence = np.clip(new_det.confidence - 0.01, 0, 1)

        # Update
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               new_det.cxcywh, new_det.confidence)
        self.scene_mean, self.scene_covariance = self.unsecented_kalman_filter.update(self.mean, self.covariance, new_det.cxcywh)
        self.obs_history.append([frame_id, new_det.cxcywh.copy(), new_det.confidence, new_det.curr_feat.copy(),
                                 self.mean.copy(), self.covariance.copy()])

        self.obs_scene_history.append([frame_id, new_det.cxcywh.copy(), new_det.confidence, new_det.curr_scene_feat.copy(),
                                 self.scene_mean.copy(), self.scene_covariance.copy()])

        if new_det.curr_feat is not None:
            self.update_features(new_det.curr_feat)
        if new_det.curr_scene_feat is not None:
            self.update_scene_features(new_det.curr_scene_feat)

        # Set
        self.is_activated = True
        self.frame_id = frame_id
        self.confidence = new_det.confidence
        self.state = TrackState.Tracked
        self.track_id = self.next_id() if new_id else self.track_id

    # 根据不同选择方式从目标特征的历史记录中选择对应的目标特征
    # 选择方式包括：从轨道的每个历史中置信度最高的检测中提取的特征，
    # 从最后一次检测（最后）中提取特征，对过去的每个特征进行平均（平均），
    # 用置信分数（W. Avg）对过去的特征进行加权平均，用指数移动平均方案（EMA）平滑过去的特征
    def get_feature(self, mode='ema'):
        # Default use smoothed feature
        feat = self.smooth_feat

        # Other options
        if mode == 'best':
            feat = self.obs_history[np.argmax(np.array([o[2] for o in self.obs_history]))][3]
        elif mode == 'last':
            feat = self.obs_history[-1][3]
        elif mode == 'avg':
            feat = np.mean(np.array([o[3] for o in self.obs_history]), axis=0)
        elif mode == 'weighted_avg':
            feat = np.sum(np.array([o[3] * o[2] for o in self.obs_history]), axis=0)
            feat /= np.sum([o[2] for o in self.obs_history])
        elif mode == 'all':
            feat = np.array([o[3] for o in self.obs_history])
            feat = feat[np.newaxis] if len(feat.shape) == 1 else feat

        return feat

    def get_scene_feature(self, mode='ema'):
        # Default use smoothed feature
        feat = self.smooth_scene_feat

        # Other options
        if mode == 'best':
            feat = self.obs_scene_history[np.argmax(np.array([o[2] for o in self.obs_scene_history]))][3]
        elif mode == 'last':
            feat = self.obs_scene_history[-1][3]
        elif mode == 'avg':
            feat = np.mean(np.array([o[3] for o in self.obs_scene_history]), axis=0)
        elif mode == 'weighted_avg':
            feat = np.sum(np.array([o[3] * o[2] for o in self.obs_scene_history]), axis=0)
            feat /= np.sum([o[2] for o in self.obs_scene_history])
        elif mode == 'all':
            feat = np.array([o[3] for o in self.obs_scene_history])
            feat = feat[np.newaxis] if len(feat.shape) == 1 else feat

        return feat

    @property
    def tlwh(self):
        """
            Get current position in bounding box format `(top left x, top left y, width, height)`.
        """
        if self.mean is None:
            x1 = self.cxcywh[0] - self.cxcywh[2] / 2
            y1 = self.cxcywh[1] - self.cxcywh[3] / 2
            w1 = self.cxcywh[2]
            h1 = self.cxcywh[3]
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            w1 = self.mean[2]
            h1 = self.mean[3]

        if self.scene_mean is None:
            x2 = self.cxcywh[0] - self.cxcywh[2] / 2
            y2 = self.cxcywh[1] - self.cxcywh[3] / 2
            w2 = self.cxcywh[2]
            h2 = self.cxcywh[3]
        else:
            x2 = self.scene_mean[0] - self.scene_mean[2] / 2
            y2 = self.scene_mean[1] - self.scene_mean[3] / 2
            w2 = self.scene_mean[2]
            h2 = self.scene_mean[3]

        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = (w1 + w2) / 2
        h = (h1 + h2) / 2


        return np.array([x, y, w, h])

    @property
    def x1y1x2y2(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
