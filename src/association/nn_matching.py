import numpy as np


def cosine_distance(a, b):
    """Pairwise cosine distance between rows of a [M,D] and b [N,D]."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-7)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-7)
    return 1.0 - a @ b.T


def nn_cosine_distance(tracks, detections, features, track_indices, detection_indices):
    """
    Cost matrix using nearest-neighbour cosine distance over each track's
    stored appearance gallery.  Falls back to 1.0 when either side lacks features.
    """
    cost = np.ones((len(track_indices), len(detection_indices)))

    valid_cols = [
        (col, detection_indices[col])
        for col in range(len(detection_indices))
        if features[detection_indices[col]] is not None
    ]
    if not valid_cols:
        return cost

    det_feats = np.array([features[didx] for _, didx in valid_cols])

    for row, tidx in enumerate(track_indices):
        if not tracks[tidx].features:
            continue
        track_feats = np.array(tracks[tidx].features)
        dists = cosine_distance(track_feats, det_feats)  # (gallery_size, n_valid_dets)
        min_dists = dists.min(axis=0)
        for k, (col, _) in enumerate(valid_cols):
            cost[row, col] = min_dists[k]

    return cost
