"""Split a full recording into rallies from the ball track."""

import numpy as np


def segment_rallies(
    xs: np.ndarray,
    ys: np.ndarray,
    fps: float,
    max_gap_s: float = 1.5,
    min_rally_s: float = 1.5,
    pad_s: float = 0.5,
    min_speed: float = 1.0,
    max_step: int = 3,
) -> list[tuple[int, int]]:
    """Return (start_frame, end_frame) spans where the ball is in play.

    xs/ys are per-frame ball positions in pixels with NaN where the ball was not
    detected. A frame is "live" when the ball is visible and moving faster than
    `min_speed` px/frame, measured over at most `max_step` frames (a ball held
    or lying on the floor is not play). Live
    frames closer than `max_gap_s` are merged, spans shorter than `min_rally_s`
    are dropped, and the rest are padded by `pad_s` without overlapping.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    visible = np.flatnonzero(~np.isnan(xs) & ~np.isnan(ys))
    if len(visible) < 2:
        return []

    step = np.diff(visible)
    speed = np.hypot(np.diff(xs[visible]), np.diff(ys[visible])) / step
    # A ball reappearing somewhere else after a gap is not evidence of motion.
    live = visible[1:][(speed >= min_speed) & (step <= max_step)]
    if len(live) == 0:
        return []

    max_gap = max_gap_s * fps
    breaks = np.flatnonzero(np.diff(live) > max_gap)
    starts = np.r_[live[0], live[breaks + 1]]
    ends = np.r_[live[breaks], live[-1]]

    keep = (ends - starts) >= min_rally_s * fps
    starts, ends = starts[keep], ends[keep]

    pad = int(round(pad_s * fps))
    n = len(xs)
    spans: list[tuple[int, int]] = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        lo = max(0, s - pad)
        hi = min(n - 1, e + pad)
        if spans and lo <= spans[-1][1]:
            # Split the dead time between neighbouring rallies down the middle.
            mid = (int(ends[i - 1]) + int(s)) // 2
            spans[-1] = (spans[-1][0], mid)
            lo = mid + 1
        spans.append((int(lo), int(hi)))
    return spans
