"""Turn per-frame ball candidates into one track by motion continuity."""

import csv
from pathlib import Path

import numpy as np

Candidate = tuple[float, float, float]  # x, y, conf


def select_track(
    candidates: dict[int, list[Candidate]],
    n_frames: int,
    strong: float = 0.5,
    max_jump: float = 150.0,
    near: float = 40.0,
    memory: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick at most one candidate per frame; returns xs, ys with NaN where none was kept.

    A strong candidate is kept unless it would mean the ball moved more than
    `max_jump` px/frame from a recent position. A weak one is kept only when it
    lies within `near` px of where the recent motion says the ball should be.
    """
    xs, ys = np.full(n_frames, np.nan), np.full(n_frames, np.nan)
    history: list[tuple[int, float, float]] = []  # last accepted (frame, x, y)
    for frame in range(n_frames):
        cands = candidates.get(frame)
        if not cands:
            continue
        history = [h for h in history if frame - h[0] <= memory]
        if not history:
            x, y, conf = max(cands, key=lambda c: c[2])
            if conf < strong:
                continue
        else:
            f1, x1, y1 = history[-1]
            if len(history) > 1:
                f0, x0, y0 = history[-2]
                vx, vy = (x1 - x0) / (f1 - f0), (y1 - y0) / (f1 - f0)
            else:
                vx = vy = 0.0
            px, py = x1 + vx * (frame - f1), y1 + vy * (frame - f1)
            x, y, conf = min(cands, key=lambda c: np.hypot(c[0] - px, c[1] - py))
            if conf >= strong:
                if np.hypot(x - x1, y - y1) > max_jump * (frame - f1):
                    continue
            elif len(history) < 2 or np.hypot(x - px, y - py) > near:
                continue
        xs[frame], ys[frame] = x, y
        history = (history + [(frame, x, y)])[-2:]
    return xs, ys


def drop_static(xs: np.ndarray, ys: np.ndarray, window: int = 6, min_travel: float = 12.0) -> None:
    """Blank detections that barely move: a stationary object mistaken for the ball. In place."""
    visible = np.flatnonzero(~np.isnan(xs))
    static = np.zeros(len(xs), dtype=bool)
    for i in range(len(visible) - window + 1):
        run = visible[i : i + window]
        if run[-1] - run[0] > 2 * window:
            continue
        if np.ptp(xs[run]) + np.ptp(ys[run]) < min_travel:
            static[run] = True
    xs[static] = ys[static] = np.nan


def drop_fragments(
    xs: np.ndarray,
    ys: np.ndarray,
    conf: np.ndarray | None = None,
    min_points: int = 5,
    max_hole: int = 3,
    min_conf: float = 0.6,
) -> None:
    """Blank isolated runs that are too short or too weak to be the ball. In place.

    A run is weak when the 80th percentile of its confidences is below
    `min_conf`: hands and heads give long but consistently hesitant runs, while
    a ball in flight scores high for most of its arc.
    """
    visible = np.flatnonzero(~np.isnan(xs))
    if len(visible) == 0:
        return
    for run in np.split(visible, np.flatnonzero(np.diff(visible) > max_hole) + 1):
        weak = conf is not None and np.percentile(conf[run], 80) < min_conf
        if len(run) < min_points or weak:
            xs[run] = ys[run] = np.nan


def load_track(
    csv_path: Path, n_frames: int, border: float = 10.0, width: float = 1280, **select
) -> tuple[np.ndarray, np.ndarray]:
    """Read a candidates CSV (frame,x,y,conf) and return the selected, cleaned per-frame track.

    Detections within `border` px of the top or sides are dropped: a ball half
    out of frame has a biased centre that looks like a change of speed.
    """
    candidates: dict[int, list[Candidate]] = {}
    with Path(csv_path).open() as f:
        for row in csv.DictReader(f):
            candidates.setdefault(int(row["frame"]), []).append(
                (float(row["x"]), float(row["y"]), float(row["conf"]))
            )
    xs, ys = select_track(candidates, n_frames, **select)
    edge = (ys < border) | (xs < border) | (xs > width - border)
    xs[edge] = ys[edge] = np.nan
    drop_static(xs, ys)
    conf = np.zeros(n_frames)
    for frame in np.flatnonzero(~np.isnan(xs)):
        conf[frame] = next(c[2] for c in candidates[frame] if c[0] == xs[frame] and c[1] == ys[frame])
    drop_fragments(xs, ys, conf)
    return xs, ys
