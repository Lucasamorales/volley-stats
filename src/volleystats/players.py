"""Player detection, pose and tracking with a YOLO pose model and BoT-SORT.

One pass gives boxes, 17 COCO keypoints and track ids for every person.

Usage: python -m volleystats.players VIDEO [START_FRAME END_FRAME]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .schema import PlayerObs
from .store import video_dir

TRACKER = str(Path(__file__).with_name("botsort_static.yaml"))
LEFT_WRIST, RIGHT_WRIST = 9, 10  # COCO keypoint indices
MIN_KEYPOINT_CONF = 0.3


def track_players(
    video: Path,
    out: Path,
    start: int = 0,
    end: int | None = None,
    weights: str = "weights/yolo26m-pose.pt",
    imgsz: int = 1280,
) -> int:
    """Track people over frames [start, end) and save them to an .npz. Returns rows written."""
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    end = end or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames, ids, boxes, confs, kpts = [], [], [], [], []
    for frame in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        res = model.track(img, persist=True, tracker=TRACKER, imgsz=imgsz, conf=0.1, verbose=False)[0]
        if res.boxes.id is None:
            continue
        n = len(res.boxes)
        frames.append(np.full(n, frame))
        ids.append(res.boxes.id.cpu().numpy().astype(int))
        boxes.append(res.boxes.xyxy.cpu().numpy())
        confs.append(res.boxes.conf.cpu().numpy())
        kpts.append(res.keypoints.data.cpu().numpy())
        if (frame - start) % 900 == 0:
            print(f"{frame - start}/{end - start} frames", flush=True)
    cap.release()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        frame=np.concatenate(frames),
        track=np.concatenate(ids),
        box=np.concatenate(boxes),
        conf=np.concatenate(confs),
        kpts=np.concatenate(kpts),
    )
    return sum(map(len, frames))


def load_players(path: Path) -> dict[int, list[PlayerObs]]:
    """Read saved tracks as frame -> players, keeping only confidently located wrists."""
    d = np.load(path)
    by_frame: dict[int, list[PlayerObs]] = {}
    for frame, track, box, k in zip(d["frame"], d["track"], d["box"], d["kpts"]):
        wrists = [tuple(map(float, k[i, :2])) for i in (LEFT_WRIST, RIGHT_WRIST) if k[i, 2] >= MIN_KEYPOINT_CONF]
        obs = PlayerObs(frame=int(frame), track_id=int(track), box=tuple(map(float, box)), wrists=wrists)
        by_frame.setdefault(int(frame), []).append(obs)
    return by_frame


if __name__ == "__main__":
    video = Path(sys.argv[1])
    start, end = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, None)
    name = "players.npz" if end is None else f"players_{start}_{end}.npz"
    print("rows:", track_players(video, video_dir(video) / name, start, end))
