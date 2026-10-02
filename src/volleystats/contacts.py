"""Find ball contacts as breaks in free flight.

Between touches the ball follows a ballistic arc: horizontal image velocity is
roughly constant and vertical velocity changes smoothly under gravity. A touch
shows up as a jump in instantaneous velocity that gravity cannot explain.

The ball is often not detected at the touch itself (it is low, blurred or
hidden by the player), so velocities are fitted to the nearest detections on
each side and extrapolated into the gap.
"""

import numpy as np

MAX_HOLE = 3  # frames of missing detections tolerated inside one fitted run
SAME_ARC_RMS = 5.0  # px; below this, one arc fits the detections on both sides of a gap

def _fit(t: int, frames: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """Polynomials for x and y as functions of (frame - t), fitted to one side of t."""
    dt = frames - t
    # A quadratic absorbs gravity; with too few points fall back to a line.
    deg = 2 if len(dt) >= 4 else 1
    return np.poly1d(np.polyfit(dt, xs, 1)), np.poly1d(np.polyfit(dt, ys, deg))


def _sides(t: int, visible: np.ndarray, k: int, span: int) -> tuple[np.ndarray, np.ndarray]:
    """The last k detections at or before t and the first k at or after it, within span frames."""
    i = np.searchsorted(visible, t)
    before = visible[max(0, i - k + 1) : i + 1] if i < len(visible) and visible[i] == t else visible[max(0, i - k) : i]
    after = visible[i : i + k]
    before, after = before[before >= t - span], after[after <= t + span]
    # A side must be one unbroken run: points beyond a hole may belong to another arc.
    holes = np.flatnonzero(np.diff(before) > MAX_HOLE)
    if len(holes):
        before = before[holes[-1] + 1 :]
    holes = np.flatnonzero(np.diff(after) > MAX_HOLE)
    if len(holes):
        after = after[: holes[0] + 1]
    return before, after


def _mismatch(t: int, visible: np.ndarray, xs: np.ndarray, ys: np.ndarray, k: int, span: int):
    """(velocity jump, position gap, mean position) between the arcs before and after t."""
    before, after = _sides(t, visible, k, span)
    if len(before) < 3 or len(after) < 3:
        return None
    bx, by = _fit(t, before, xs[before], ys[before])
    ax, ay = _fit(t, after, xs[after], ys[after])
    if after[0] - before[-1] > MAX_HOLE:
        # Across a detection gap (ball above the frame or hidden): if one arc
        # explains both sides, the ball was simply in flight.
        both = np.r_[before, after]
        jx, jy = _fit(t, both, xs[both], ys[both])
        rms = np.sqrt(np.mean((jx(both - t) - xs[both]) ** 2 + (jy(both - t) - ys[both]) ** 2))
        if rms < SAME_ARC_RMS:
            return 0.0, 0.0, (jx(0), jy(0))
    jump = np.hypot(ax.deriv()(0) - bx.deriv()(0), ay.deriv()(0) - by.deriv()(0))
    gap = np.hypot(ax(0) - bx(0), ay(0) - by(0))
    return float(jump), float(gap), ((ax(0) + bx(0)) / 2, (ay(0) + by(0)) / 2)


def velocity_change(xs: np.ndarray, ys: np.ndarray, k: int = 6, span: int = 20) -> np.ndarray:
    """Per-frame size of the velocity jump (px/frame); NaN where it cannot be computed."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    visible = np.flatnonzero(~np.isnan(xs) & ~np.isnan(ys))
    score = np.full(len(xs), np.nan)
    if len(visible) < 6:
        return score
    for t in range(visible[0], visible[-1] + 1):
        m = _mismatch(t, visible, xs, ys, k, span)
        if m is not None:
            score[t] = m[0]
    return score


def ball_at(xs: np.ndarray, ys: np.ndarray, t: int, k: int = 6, span: int = 20) -> tuple[float, float]:
    """Ball position at frame t: the detection if there is one, else where the two arcs meet."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if not np.isnan(xs[t]):
        return float(xs[t]), float(ys[t])
    visible = np.flatnonzero(~np.isnan(xs) & ~np.isnan(ys))
    m = _mismatch(t, visible, xs, ys, k, span)
    if m is None:
        nearest = visible[np.argmin(np.abs(visible - t))]
        return float(xs[nearest]), float(ys[nearest])
    return float(m[2][0]), float(m[2][1])


def detect_contacts(
    xs: np.ndarray,
    ys: np.ndarray,
    k: int = 6,
    span: int = 20,
    min_change: float = 8.0,
    min_gap: int = 8,
) -> list[tuple[int, float]]:
    """Return (frame, score) for each contact, in frame order.

    `min_change` is the velocity jump in px/frame that counts as a touch and
    `min_gap` the minimum number of frames between two touches. Both depend on
    resolution and frame rate and need tuning on real footage.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    visible = np.flatnonzero(~np.isnan(xs) & ~np.isnan(ys))
    score = velocity_change(xs, ys, k, span)
    candidates = np.flatnonzero(np.nan_to_num(score) >= min_change)
    picked: dict[int, float] = {}
    # Strongest first, suppressing weaker neighbours of the same touch.
    for t in candidates[np.argsort(-score[candidates])]:
        refined = _refine(int(t), visible, xs, ys, k, span)
        if all(abs(refined - p) >= min_gap for p in picked):
            picked[refined] = float(score[t])
    return sorted(picked.items())


def _refine(t: int, visible: np.ndarray, xs: np.ndarray, ys: np.ndarray, k: int, span: int) -> int:
    """Inside a detection gap, move t to the frame where the two arcs come closest."""
    if not np.isnan(xs[t]):
        return t
    i = np.searchsorted(visible, t)
    lo, hi = visible[i - 1], visible[i]
    gaps = {u: _mismatch(u, visible, xs, ys, k, span) for u in range(lo + 1, hi)}
    gaps = {u: m[1] for u, m in gaps.items() if m is not None}
    return min(gaps, key=gaps.get) if gaps else t
