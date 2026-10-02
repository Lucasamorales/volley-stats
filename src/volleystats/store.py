"""Where per-video outputs live, and loading/saving the match record."""

from pathlib import Path

import cv2

from .schema import MatchRecord, VideoMeta

PROCESSED = Path("data/processed")


def video_dir(video: Path) -> Path:
    out = PROCESSED / Path(video).stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def probe(video: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    meta = VideoMeta(
        path=str(video),
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def load_match(video: Path) -> MatchRecord:
    """The saved record for this video, or a fresh one."""
    path = video_dir(video) / "match.json"
    if path.exists():
        return MatchRecord.model_validate_json(path.read_text())
    return MatchRecord(video=probe(video))


def save_match(video: Path, match: MatchRecord) -> Path:
    path = video_dir(video) / "match.json"
    path.write_text(match.model_dump_json(indent=2))
    return path
