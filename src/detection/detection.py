import numpy as np


class Detection:
    """Bounding box detection in [x, y, w, h] (top-left + dims) format."""

    def __init__(self, tlwh, confidence, class_id=0):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.class_id = int(class_id)

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xywh(self):
        """[cx, cy, w, h]"""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xyah(self):
        """[cx, cy, aspect_ratio, height]"""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
