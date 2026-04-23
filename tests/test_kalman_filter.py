import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.tracking.kalman_filter import KalmanFilter


@pytest.fixture
def kf():
    return KalmanFilter()


@pytest.fixture
def init_state(kf):
    m = np.array([100.0, 200.0, 50.0, 80.0])
    return kf.initiate(m), m


def test_initiate_shapes(kf, init_state):
    (mean, cov), m = init_state
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    np.testing.assert_array_equal(mean[:4], m)
    np.testing.assert_array_equal(mean[4:], np.zeros(4))


def test_predict_shapes(kf, init_state):
    (mean, cov), _ = init_state
    pred_mean, pred_cov = kf.predict(mean, cov)
    assert pred_mean.shape == (8,)
    assert pred_cov.shape == (8, 8)


def test_update_shapes(kf, init_state):
    (mean, cov), m = init_state
    pred_mean, pred_cov = kf.predict(mean, cov)
    new_mean, new_cov = kf.update(pred_mean, pred_cov, m)
    assert new_mean.shape == (8,)
    assert new_cov.shape == (8, 8)


def test_gating_distance_ordering(kf, init_state):
    (mean, cov), m = init_state
    close = m.copy()
    far = m + np.array([500.0, 500.0, 0.0, 0.0])
    dists = kf.gating_distance(mean, cov, np.array([close, far]))
    assert dists.shape == (2,)
    assert dists[0] < dists[1]


def test_covariance_grows_on_predict(kf, init_state):
    (mean, cov), _ = init_state
    for _ in range(5):
        mean, cov = kf.predict(mean, cov)
    # After several predictions without updates, trace should be larger
    (_, init_cov), _ = init_state
    assert np.trace(cov) > np.trace(init_cov)
