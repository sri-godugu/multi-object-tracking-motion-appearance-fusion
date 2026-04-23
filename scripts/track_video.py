"""
Main tracking script.

Usage:
    python scripts/track_video.py --source video.mp4 --output out.mp4
    python scripts/track_video.py --source 0 --display          # webcam
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
from src.utils.io_utils import VideoReader, VideoWriter
from src.utils.visualization import draw_tracks


def parse_args():
    p = argparse.ArgumentParser(description="DeepSORT-style Multi-Object Tracker")
    p.add_argument("--source", required=True, help="Video path or camera index")
    p.add_argument("--output", default=None, help="Output video path (mp4)")
    p.add_argument("--detector", default="yolov8n.pt", help="YOLOv8 weights file")
    p.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--classes", nargs="+", type=int, default=[0], help="COCO class IDs (0=person)")
    p.add_argument("--max-cosine-dist", type=float, default=0.3)
    p.add_argument("--max-iou-dist", type=float, default=0.7)
    p.add_argument("--max-age", type=int, default=30, help="Frames before deleting a lost track")
    p.add_argument("--n-init", type=int, default=3, help="Hits to confirm a tentative track")
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    p.add_argument("--display", action="store_true", help="Show live preview window")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Device: {device}")

    detector = Detector(
        model_path=args.detector,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        classes=args.classes,
        device=device,
    )
    extractor = FeatureExtractor(device=device)
    tracker = Tracker(
        max_cosine_distance=args.max_cosine_dist,
        max_iou_distance=args.max_iou_dist,
        max_age=args.max_age,
        n_init=args.n_init,
    )

    with VideoReader(source) as reader:
        writer = VideoWriter(args.output, reader.fps, reader.width, reader.height) \
            if args.output else None
        frame_idx = 0
        try:
            for frame in reader:
                # ── Detection ──────────────────────────────────────────
                detections = detector.detect(frame)

                # ── Feature Extraction ─────────────────────────────────
                features = extractor.extract(frame, detections)

                # ── Predict + Associate + Track Update ─────────────────
                tracker.predict()
                tracker.update(detections, features)

                # ── Visualise ──────────────────────────────────────────
                vis = frame.copy()
                draw_tracks(vis, tracker.tracks)

                frame_idx += 1
                n_confirmed = sum(1 for t in tracker.tracks if t.is_confirmed())
                print(f"\rFrame {frame_idx:5d} | Active tracks: {n_confirmed:3d}", end="", flush=True)

                if writer:
                    writer.write(vis)
                if args.display:
                    cv2.imshow("Tracking", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if writer:
                writer.release()
            cv2.destroyAllWindows()

    print(f"\nDone — processed {frame_idx} frames.")


if __name__ == "__main__":
    main()
