import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.association.linear_assignment import min_cost_matching, matching_cascade


def cost_2x2(tracks, detections, features, track_indices, detection_indices):
    return np.array([[0.1, 0.9], [0.8, 0.2]])


def test_min_cost_perfect_matching():
    tracks = [None, None]
    detections = [None, None]
    features = [None, None]
    matches, ut, ud = min_cost_matching(cost_2x2, 0.5, tracks, detections, features)
    assert set(matches) == {(0, 0), (1, 1)}
    assert ut == []
    assert ud == []


def test_min_cost_threshold_rejection():
    # All costs > max_distance → everything unmatched
    def high_cost(t, d, f, ti, di):
        return np.array([[0.9, 0.95], [0.85, 0.92]])

    tracks = [None, None]
    detections = [None, None]
    features = [None, None]
    matches, ut, ud = min_cost_matching(high_cost, 0.5, tracks, detections, features)
    assert matches == []
    assert len(ut) == 2
    assert len(ud) == 2


def test_empty_inputs():
    matches, ut, ud = min_cost_matching(cost_2x2, 0.5, [], [], [], [], [])
    assert matches == [] and ut == [] and ud == []


def test_matching_cascade_by_age():
    """Tracks with time_since_update==1 should be matched first."""

    class FakeTrack:
        def __init__(self, age):
            self.time_since_update = age
            self.features = []

    tracks = [FakeTrack(1), FakeTrack(2)]
    detections = [None, None]
    features = [None, None]

    def identity_cost(t, d, f, ti, di):
        return np.zeros((len(ti), len(di)))

    matches, _, _ = matching_cascade(
        identity_cost, 0.5, 3, tracks, detections, features
    )
    assert len(matches) >= 1
