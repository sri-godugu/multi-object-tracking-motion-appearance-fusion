"""
Run the tracker on a MOT-format dataset and write results for TrackEval / py-motmetrics.

Expected sequence layout:
  <data-dir>/
    <seq-name>/
      img1/  (000001.jpg, 000002.jpg, …)
      gt/gt.txt
      seqinfo.ini

Output (MOT challenge format):
  <output-dir>/<seq-name>.txt
  Columns: frame, id, x, y, w, h, conf, -1, -1, -1
"""

import argparse
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.detector import Detector
from src.features.extractor import FeatureExtractor
from src.tracking.tracker import Tracker
from src.tracking.track import Track


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Root of MOT dataset")
    p.add_argument("--output-dir", default="results", help="Where to write .txt results")
    p.add_argument("--detector", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default=None)
    return p.parse_args()


def run_sequence(seq_path, detector, extractor):
    img_dir = os.path.join(seq_path, "img1")
    frames = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))

    Track._id_counter = 0  # reset per sequence so IDs start at 1
    tracker = Tracker(max_cosine_distance=0.3, max_iou_distance=0.7, max_age=30, n_init=3)
    rows = []

    for frame_idx, fname in enumerate(frames, 1):
        frame = cv2.imread(os.path.join(img_dir, fname))
        detections = detector.detect(frame)
        features = extractor.extract(frame, detections)
        tracker.predict()
        tracker.update(detections, features)

        for track in tracker.tracks:
            if not track.is_confirmed():
                continue
            x, y, w, h = track.to_tlwh()
            rows.append(f"{frame_idx},{track.track_id},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,-1,-1,-1")

    return rows


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    detector = Detector(args.detector, conf_threshold=args.conf, device=device)
    extractor = FeatureExtractor(device=device)
    os.makedirs(args.output_dir, exist_ok=True)

    sequences = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )

    for seq_name in sequences:
        seq_path = os.path.join(args.data_dir, seq_name)
        if not os.path.isdir(os.path.join(seq_path, "img1")):
            continue
        print(f"Processing {seq_name} ...", flush=True)
        rows = run_sequence(seq_path, detector, extractor)
        out_path = os.path.join(args.output_dir, f"{seq_name}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"  → {len(rows)} track lines saved to {out_path}")

    print("Evaluation complete. Feed results to TrackEval or py-motmetrics.")


if __name__ == "__main__":
    main()
