import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.tracking.tracker import Tracker
from src.tracking.track import Track
from src.detection.detection import Detection


def _det(x=100, y=200, w=50, h=80, conf=0.9):
    return Detection([x, y, w, h], conf, class_id=0)


@pytest.fixture(autouse=True)
def reset_track_counter():
    Track._id_counter = 0
    yield
    Track._id_counter = 0


def test_empty_update():
    tracker = Tracker()
    tracker.predict()
    tracker.update([])
    assert tracker.tracks == []


def test_single_track_confirmed_after_n_init():
    tracker = Tracker(n_init=3)
    det = _det()
    for _ in range(3):
        tracker.predict()
        tracker.update([det])
    confirmed = [t for t in tracker.tracks if t.is_confirmed()]
    assert len(confirmed) == 1


def test_single_track_confirmed_n_init_1():
    tracker = Tracker(n_init=1)
    tracker.predict()
    tracker.update([_det()])
    assert tracker.tracks[0].is_confirmed()


def test_unique_track_ids():
    tracker = Tracker(n_init=1)
    tracker.predict()
    tracker.update([_det(100, 200), _det(400, 200)])
    ids = [t.track_id for t in tracker.tracks]
    assert len(ids) == len(set(ids))


def test_track_deleted_after_max_age():
    tracker = Tracker(n_init=1, max_age=3)
    tracker.predict()
    tracker.update([_det()])
    assert len(tracker.tracks) == 1

    for _ in range(4):      # 4 > max_age=3
        tracker.predict()
        tracker.update([])  # no detections → track ages out

    assert len(tracker.tracks) == 0


def test_track_count_with_features():
    tracker = Tracker(n_init=1)
    import numpy as np
    det = _det()
    feat = np.random.randn(128).astype("float32")
    feat /= (feat ** 2).sum() ** 0.5
    tracker.predict()
    tracker.update([det], [feat])
    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0].features) == 1
