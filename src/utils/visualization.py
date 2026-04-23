import colorsys
import random
import cv2


def _track_color(track_id):
    """Deterministic per-ID colour (BGR)."""
    random.seed(track_id)
    h = random.random()
    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(frame, tracks, show_tentative=False):
    for track in tracks:
        if not show_tentative and not track.is_confirmed():
            continue
        x1, y1, x2, y2 = track.to_tlbr().astype(int)
        color = _track_color(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track.track_id}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_detections(frame, detections, color=(0, 255, 0)):
    for det in detections:
        x1, y1, x2, y2 = det.to_tlbr().astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, f"{det.confidence:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame
