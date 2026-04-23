import numpy as np


class KalmanFilter:
    """
    Kalman filter for axis-aligned bounding boxes.

    State  : [cx, cy, w, h, vcx, vcy, vw, vh]
    Measure: [cx, cy, w, h]

    Uncertainty weights follow the DeepSORT convention: position noise is
    proportional to bounding-box width/height so the filter adapts to scale.
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Constant-velocity state transition
        self.F = np.eye(2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = dt

        # Measurement picks out position components
        self.H = np.eye(ndim, 2 * ndim)

        self._std_weight_pos = 1.0 / 20
        self._std_weight_vel = 1.0 / 160

    def initiate(self, measurement):
        """Bootstrap mean and covariance from a single measurement [cx,cy,w,h]."""
        mean = np.concatenate([measurement, np.zeros(4)])
        w, h = measurement[2], measurement[3]
        std = [
            2 * self._std_weight_pos * w,
            2 * self._std_weight_pos * h,
            2 * self._std_weight_pos * w,
            2 * self._std_weight_pos * h,
            10 * self._std_weight_vel * w,
            10 * self._std_weight_vel * h,
            10 * self._std_weight_vel * w,
            10 * self._std_weight_vel * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        w, h = mean[2], mean[3]
        std_pos = [self._std_weight_pos * s for s in [w, h, w, h]]
        std_vel = [self._std_weight_vel * s for s in [w, h, w, h]]
        Q = np.diag(np.square(std_pos + std_vel))
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q
        return mean, covariance

    def project(self, mean, covariance):
        w, h = mean[2], mean[3]
        std = [self._std_weight_pos * s for s in [w, h, w, h]]
        R = np.diag(np.square(std))
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T + R
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        proj_mean, proj_cov = self.project(mean, covariance)
        K = covariance @ self.H.T @ np.linalg.inv(proj_cov)
        mean = mean + K @ (measurement - proj_mean)
        covariance = covariance - K @ proj_cov @ K.T
        return mean, covariance

    def gating_distance(self, mean, covariance, measurements):
        """Squared Mahalanobis distance between projected state and measurements."""
        proj_mean, proj_cov = self.project(mean, covariance)
        diff = measurements - proj_mean
        chol = np.linalg.cholesky(proj_cov)
        z = np.linalg.solve(chol, diff.T)
        return np.sum(z * z, axis=0)
