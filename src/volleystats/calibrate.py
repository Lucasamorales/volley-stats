"""Court calibration: click court landmarks on a frame and save the homography.

Usage: python -m volleystats.calibrate VIDEO

For each landmark named at the top of the window, click it on the image, or
press S to skip one that is hidden or out of frame, U to undo, Q to quit
without saving. At least four landmarks are needed.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from . import court
from .schema import MatchRecord
from .store import load_match, save_match


def set_court(match: MatchRecord, points: dict[str, tuple[float, float]], estimated: bool) -> None:
    """Store clicked landmarks and the homography fitted to them."""
    image = np.array(list(points.values()), dtype=float)
    metres = np.array([court.LANDMARKS[name] for name in points])
    match.court_points = points
    match.court_estimated = estimated
    match.homography = court.fit_homography(image, metres).tolist()


def draw_court(frame: np.ndarray, H: np.ndarray, colour=(0, 0, 255)) -> np.ndarray:
    """Overlay the court lines implied by homography H (image -> court)."""
    inv = np.linalg.inv(H)
    lines = [
        ((0, 0), (court.WIDTH, 0)),
        ((court.WIDTH, 0), (court.WIDTH, court.LENGTH)),
        ((court.WIDTH, court.LENGTH), (0, court.LENGTH)),
        ((0, court.LENGTH), (0, 0)),
        ((0, court.NET_Y), (court.WIDTH, court.NET_Y)),
        ((0, 6.0), (court.WIDTH, 6.0)),
        ((0, 12.0), (court.WIDTH, 12.0)),
    ]
    for a, b in lines:
        pa, pb = court.to_court(inv, [a, b])
        cv2.line(frame, tuple(map(int, pa)), tuple(map(int, pb)), colour, 2)
    return frame


def click_landmarks(frame: np.ndarray) -> dict[str, tuple[float, float]] | None:
    names = list(court.LANDMARKS)
    points: dict[str, tuple[float, float]] = {}
    skipped: list[str] = []
    window = "court calibration"

    def current() -> str | None:
        return next((n for n in names if n not in points and n not in skipped), None)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and current():
            points[current()] = (float(x), float(y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        view = frame.copy()
        for name, (x, y) in points.items():
            cv2.circle(view, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.putText(view, name, (int(x) + 8, int(y) - 8), 0, 0.5, (0, 255, 255), 1)
        if len(points) >= 4:
            metres = np.array([court.LANDMARKS[n] for n in points])
            draw_court(view, court.fit_homography(np.array(list(points.values())), metres))
        prompt = f"click: {current()}  (S skip, U undo)" if current() else "ENTER to save, U undo"
        cv2.putText(view, prompt, (20, 40), 0, 1.0, (255, 255, 255), 3)
        cv2.putText(view, prompt, (20, 40), 0, 1.0, (0, 0, 0), 1)
        cv2.imshow(window, view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            points = None
            break
        if key == ord("s") and current():
            skipped.append(current())
        if key == ord("u"):
            done = [n for n in names if n in points or n in skipped]
            if done:
                points.pop(done[-1], None)
                if done[-1] in skipped:
                    skipped.remove(done[-1])
        if key in (13, 10) and len(points) >= 4:
            break
    cv2.destroyAllWindows()
    return points


if __name__ == "__main__":
    video = Path(sys.argv[1])
    match = load_match(video)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, match.video.n_frames // 2)
    ok, frame = cap.read()
    clicked = click_landmarks(frame)
    if clicked:
        set_court(match, clicked, estimated=False)
        print("saved", save_match(video, match))
