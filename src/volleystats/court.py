"""Court geometry: homography from clicked points, and which half a player stands in.

Court coordinates are metres: x across the court (0..9), y along it (0..18),
with y=0 the end line nearest the camera and the net at y=9.
"""

import numpy as np

from .schema import Side, Team

WIDTH = 9.0
LENGTH = 18.0
NET_Y = 9.0

# Landmarks the user can click; any four or more (not all collinear) fix the homography.
LANDMARKS: dict[str, tuple[float, float]] = {
    "near_left": (0.0, 0.0),
    "near_right": (WIDTH, 0.0),
    "far_right": (WIDTH, LENGTH),
    "far_left": (0.0, LENGTH),
    "net_left": (0.0, NET_Y),
    "net_right": (WIDTH, NET_Y),
    "near_attack_left": (0.0, 6.0),
    "near_attack_right": (WIDTH, 6.0),
    "far_attack_left": (0.0, 12.0),
    "far_attack_right": (WIDTH, 12.0),
}


def fit_homography(image_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    """Least-squares homography mapping image pixels to court metres (DLT)."""
    image_pts = np.asarray(image_pts, dtype=float)
    court_pts = np.asarray(court_pts, dtype=float)
    if len(image_pts) < 4 or len(image_pts) != len(court_pts):
        raise ValueError("need at least 4 matching image/court points")

    # Scale pixels to roughly unit size so the SVD is well conditioned.
    mean = image_pts.mean(axis=0)
    scale = np.sqrt(2) / np.linalg.norm(image_pts - mean, axis=1).mean()
    T = np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]])
    norm = (image_pts - mean) * scale

    rows = []
    for (u, v), (x, y) in zip(norm, court_pts):
        rows.append([-u, -v, -1, 0, 0, 0, x * u, x * v, x])
        rows.append([0, 0, 0, -u, -v, -1, y * u, y * v, y])
    _, _, vt = np.linalg.svd(np.array(rows))
    H = vt[-1].reshape(3, 3) @ T
    return H / H[2, 2]


def to_court(H: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    """Project image points lying on the floor (e.g. feet) to court metres."""
    pts = np.atleast_2d(np.asarray(image_pts, dtype=float))
    homog = np.c_[pts, np.ones(len(pts))] @ np.asarray(H).T
    return homog[:, :2] / homog[:, 2:3]


def side_of(court_xy: tuple[float, float]) -> Side:
    return Side.NEAR if court_xy[1] < NET_Y else Side.FAR


def on_court(court_xy: tuple[float, float], margin: float = 2.0) -> bool:
    """True for points on the court or within `margin` metres of it (servers, chasers)."""
    x, y = court_xy
    return -margin <= x <= WIDTH + margin and -margin <= y <= LENGTH + margin


def team_of(court_xy: tuple[float, float], us_side: Side) -> Team:
    return Team.US if side_of(court_xy) is us_side else Team.THEM
