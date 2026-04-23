from enum import Enum
import numpy as np


class TrackState(Enum):
    Tentative = 1   # not yet confirmed (< n_init hits)
    Confirmed = 2   # confirmed, actively tracked
    Deleted = 3     # lost too long; will be removed


class Track:
    """
    Single object track.

    A track starts Tentative and becomes Confirmed after n_init consecutive
    detections.  If it goes unmatched for max_age frames it is Deleted.
    Appearance features are stored as a rolling buffer (last 100 crops).
    """

    _id_counter = 0

    def __init__(self, mean, covariance, feature, n_init=3, max_age=30):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.mean = mean
        self.covariance = covariance
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self._n_init = n_init
        self._max_age = max_age
        self.features = [feature] if feature is not None else []

    # ------------------------------------------------------------------
    # Kalman predict / update
    # ------------------------------------------------------------------

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, feature):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xywh()
        )
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 100:
                self.features.pop(0)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def to_tlwh(self):
        """[x, y, w, h] (top-left origin)."""
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """[x1, y1, x2, y2]."""
        ret = self.to_tlwh()
        ret[2:] += ret[:2]
        return ret
