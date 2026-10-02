"""Draw pipeline output on video frames so a person can check it."""

from pathlib import Path

import cv2
import numpy as np

from . import court
from .schema import PlayerObs, Side

COLOURS = {Side.NEAR: (80, 220, 80), Side.FAR: (80, 160, 255), None: (150, 150, 150)}


def player_side(p: PlayerObs, H: np.ndarray | None, margin: float = 1.5) -> Side | None:
    """Which half the player stands in, or None for people off the court."""
    if H is None:
        return None
    xy = tuple(court.to_court(H, p.foot)[0])
    return court.side_of(xy) if court.on_court(xy, margin) else None


def annotate(
    img: np.ndarray,
    players: list[PlayerObs],
    ball: tuple[float, float] | None = None,
    H: np.ndarray | None = None,
    names: dict[int, str] | None = None,
    note: str = "",
) -> np.ndarray:
    for p in players:
        colour = COLOURS[player_side(p, H)]
        x1, y1, x2, y2 = map(int, p.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        label = (names or {}).get(p.track_id, str(p.track_id))
        cv2.putText(img, label, (x1, y1 - 5), 0, 0.55, (0, 0, 0), 3)
        cv2.putText(img, label, (x1, y1 - 5), 0, 0.55, colour, 1)
        for wx, wy in p.wrists:
            cv2.circle(img, (int(wx), int(wy)), 3, (255, 0, 255), -1)
    if ball is not None:
        cv2.circle(img, (int(ball[0]), int(ball[1])), 9, (0, 255, 255), 2)
    if note:
        cv2.putText(img, note, (15, 30), 0, 0.8, (0, 0, 0), 4)
        cv2.putText(img, note, (15, 30), 0, 0.8, (255, 255, 255), 2)
    return img


def render_clip(
    video: Path,
    out: Path,
    start: int,
    end: int,
    players: dict[int, list[PlayerObs]],
    xs: np.ndarray,
    ys: np.ndarray,
    H: np.ndarray | None = None,
    names: dict[int, str] | None = None,
    notes: dict[int, str] | None = None,
) -> None:
    """Write frames [start, end) with boxes, track labels, wrists and the ball."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    note = ""
    for frame in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        ball = None if np.isnan(xs[frame]) else (xs[frame], ys[frame])
        note = (notes or {}).get(frame, note)
        writer.write(annotate(img, players.get(frame, []), ball, H, names, f"{frame}  {note}"))
    writer.release()
    cap.release()
