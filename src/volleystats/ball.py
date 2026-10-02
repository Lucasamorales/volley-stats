"""Ball tracking with a VballNet heatmap model (ONNX).

The model takes 9 consecutive grayscale frames at 288x512 and returns one
heatmap per frame. Model and recipe: github.com/asigatchov/fast-volleyball-tracking-inference (MIT).

Detection writes up to three candidate blobs per frame down to a low
confidence; selection.load_track then picks one per frame by motion continuity, so the
threshold can be changed without re-running the model.

Usage: python -m volleystats.ball VIDEO MODEL.onnx OUT.csv
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SEQ = 9
IN_H, IN_W = 288, 512
FLOOR = 0.25  # weakest heatmap peak kept as a candidate
TOP_K = 3


def _candidates(heatmap: np.ndarray) -> list[tuple[float, float, float]]:
    """Blobs above FLOOR as (x, y, peak) in heatmap pixels, strongest peak first."""
    n, labels = cv2.connectedComponents((heatmap >= FLOOR).astype(np.uint8))
    blobs = []
    for label in range(1, n):
        ys, xs = np.nonzero(labels == label)
        w = heatmap[ys, xs]
        blobs.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), float(w.max())))
    return sorted(blobs, key=lambda b: -b[2])[:TOP_K]


def track_ball(video: Path, model: Path, out_csv: Path) -> int:
    """Write frame,x,y,conf for every candidate blob. Returns frames read."""
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(str(video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sx, sy = width / IN_W, height / IN_H

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = 0
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "conf"])
        while True:
            clip = []
            while len(clip) < SEQ:
                ok, img = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clip.append(cv2.resize(gray, (IN_W, IN_H)).astype(np.float32) / 255.0)
            if not clip:
                break
            n_real = len(clip)
            clip += [clip[-1]] * (SEQ - n_real)  # pad the final partial clip

            heatmaps = session.run(None, {input_name: np.stack(clip)[None]})[0][0]
            for i in range(n_real):
                for x, y, conf in _candidates(heatmaps[i]):
                    writer.writerow([frame, round(x * sx, 1), round(y * sy, 1), round(conf, 3)])
                frame += 1
            if frame % 1800 < SEQ:
                print(f"{frame}/{total} frames", flush=True)
    cap.release()
    return frame


if __name__ == "__main__":
    video, model, out = map(Path, sys.argv[1:4])
    print(f"done: {track_ball(video, model, out)} frames")
