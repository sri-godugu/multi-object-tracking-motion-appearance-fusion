from .kalman_filter import KalmanFilter
from .track import Track
from ..association import iou_matching, nn_matching, linear_assignment


class Tracker:
    """
    DeepSORT-style multi-object tracker.

    Each call to update() runs the full pipeline:
      Predict → Cascade appearance match → IoU fallback match → Track update
    """

    def __init__(self, max_cosine_distance=0.3, max_iou_distance=0.7,
                 max_age=30, n_init=3):
        self.kf = KalmanFilter()
        self.max_cosine_distance = max_cosine_distance
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.tracks = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self):
        """Advance all tracks one step with the Kalman filter."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, features=None):
        """
        Associate detections to tracks and update state.

        Parameters
        ----------
        detections : list[Detection]
        features   : list[np.ndarray | None], same length as detections
        """
        if features is None:
            features = [None] * len(detections)

        matches, unmatched_tracks, unmatched_dets = self._match(detections, features)

        for tidx, didx in matches:
            self.tracks[tidx].update(self.kf, detections[didx], features[didx])

        for tidx in unmatched_tracks:
            self.tracks[tidx].mark_missed()

        for didx in unmatched_dets:
            self._initiate_track(detections[didx], features[didx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match(self, detections, features):
        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Stage 1 – appearance cascade on confirmed tracks
        matches_a, unmatched_confirmed, unmatched_dets = linear_assignment.matching_cascade(
            nn_matching.nn_cosine_distance,
            self.max_cosine_distance,
            self.max_age,
            self.tracks,
            detections,
            features,
            confirmed,
        )

        # Stage 2 – IoU matching on unconfirmed + recently lost confirmed tracks
        iou_candidates = unconfirmed + [
            k for k in unmatched_confirmed
            if self.tracks[k].time_since_update == 1
        ]
        still_lost = [
            k for k in unmatched_confirmed
            if self.tracks[k].time_since_update != 1
        ]

        matches_b, unmatched_iou, unmatched_dets = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            features,
            iou_candidates,
            unmatched_dets,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(still_lost + unmatched_iou))
        return matches, unmatched_tracks, unmatched_dets

    def _initiate_track(self, detection, feature):
        mean, covariance = self.kf.initiate(detection.to_xywh())
        self.tracks.append(Track(mean, covariance, feature, self.n_init, self.max_age))
