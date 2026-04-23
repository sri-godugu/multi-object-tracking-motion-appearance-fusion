import numpy as np


def iou(bbox, candidates):
    """
    Compute IoU between one bbox [x,y,w,h] and N candidate boxes [N,4].
    Both in top-left + dims format.
    """
    tl = np.maximum(bbox[:2], candidates[:, :2])
    br = np.minimum(bbox[:2] + bbox[2:], candidates[:, :2] + candidates[:, 2:])
    wh = np.maximum(0.0, br - tl)
    intersection = wh[:, 0] * wh[:, 1]
    area_bbox = bbox[2] * bbox[3]
    area_cands = candidates[:, 2] * candidates[:, 3]
    return intersection / (area_bbox + area_cands - intersection + 1e-7)


def iou_cost(tracks, detections, features, track_indices, detection_indices):
    """Cost matrix using 1 – IoU; ignores appearance."""
    cost = np.zeros((len(track_indices), len(detection_indices)))
    cand_boxes = np.array([detections[i].tlwh for i in detection_indices])
    for row, tidx in enumerate(track_indices):
        cost[row] = 1.0 - iou(tracks[tidx].to_tlwh(), cand_boxes)
    return cost
