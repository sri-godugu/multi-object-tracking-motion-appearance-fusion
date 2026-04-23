import numpy as np
from scipy.optimize import linear_sum_assignment


def min_cost_matching(distance_metric, max_distance, tracks, detections, features,
                      track_indices=None, detection_indices=None):
    """
    Solve a single assignment round with the Hungarian algorithm.

    Returns
    -------
    matches            : list of (track_idx, detection_idx) pairs
    unmatched_tracks   : list of unmatched track indices
    unmatched_dets     : list of unmatched detection indices
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    if not track_indices or not detection_indices:
        return [], list(track_indices), list(detection_indices)

    cost_matrix = distance_metric(
        tracks, detections, features, track_indices, detection_indices
    )
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    row_set = set(row_ind.tolist())
    col_set = set(col_ind.tolist())

    unmatched_tracks = [track_indices[r] for r in range(len(track_indices)) if r not in row_set]
    unmatched_dets = [detection_indices[c] for c in range(len(detection_indices)) if c not in col_set]
    matches = []

    for r, c in zip(row_ind, col_ind):
        tidx = track_indices[r]
        didx = detection_indices[c]
        if cost_matrix[r, c] > max_distance:
            unmatched_tracks.append(tidx)
            unmatched_dets.append(didx)
        else:
            matches.append((tidx, didx))

    return matches, unmatched_tracks, unmatched_dets


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections,
                     features, track_indices=None, detection_indices=None):
    """
    Priority-queue cascade: match tracks that were last seen most recently first.
    Tracks unseen for 1 frame get first pick; older tracks follow.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_dets = list(detection_indices)
    all_matches = []

    for level in range(cascade_depth):
        if not unmatched_dets:
            break
        level_track_indices = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if not level_track_indices:
            continue
        matches_l, _, unmatched_dets = min_cost_matching(
            distance_metric, max_distance, tracks, detections, features,
            level_track_indices, unmatched_dets,
        )
        all_matches.extend(matches_l)

    matched_track_set = {k for k, _ in all_matches}
    unmatched_tracks = [k for k in track_indices if k not in matched_track_set]
    return all_matches, unmatched_tracks, unmatched_dets
