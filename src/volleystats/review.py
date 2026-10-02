"""Local web page for checking the touches the pipeline proposes.

Usage: python -m volleystats.review     then open http://127.0.0.1:8000

Every edit is saved to match.json at once and appended to corrections.jsonl,
which doubles as training data. The server binds to localhost only: it serves
footage of real people.
"""

import json
import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import court
from .contacts import ball_at
from .pipeline import players_file, relabel
from .players import load_players
from .schema import Action, Contact, MatchRecord, PlayerObs, Rally, Side, Team
from .selection import load_track
from .store import PROCESSED

RAW = Path("data/raw")
PAGE = Path(__file__).with_name("review.html")

app = FastAPI()
_extract_lock = threading.Lock()
_save_lock = threading.Lock()


class Edit(BaseModel):
    op: str
    frame: int | None = None
    value: str | None = None
    track_id: int | None = None


def _video(stem: str) -> Path:
    path = RAW / f"{stem}.mp4"
    if not path.exists() or not (PROCESSED / stem / "match.json").exists():
        raise HTTPException(404, f"no processed video {stem}")
    return path


def _load(stem: str) -> MatchRecord:
    _video(stem)
    return MatchRecord.model_validate_json((PROCESSED / stem / "match.json").read_text())


def _save(stem: str, match: MatchRecord) -> None:
    (PROCESSED / stem / "match.json").write_text(match.model_dump_json(indent=2))


def _log(stem: str, rally: Rally, edit: Edit, old, new) -> None:
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video": stem,
        "rally_start": rally.start_frame,
        "op": edit.op,
        "frame": edit.frame,
        "old": old,
        "new": new,
    }
    with (PROCESSED / stem / "corrections.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


@lru_cache(maxsize=8)
def _players(path: str) -> dict[int, list[PlayerObs]]:
    return load_players(Path(path))


@lru_cache(maxsize=4)
def _ball(stem: str, n_frames: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    return load_track(PROCESSED / stem / "ball.csv", n_frames, width=width)


def _us_side(match: MatchRecord, rally: Rally) -> Side:
    return next(s.us_side for s in match.sets if s.set_number == rally.set_number)


def _team_of(p: PlayerObs, H: np.ndarray, us_side: Side) -> tuple[Team | None, tuple[float, float]]:
    """Team from where the player stands in this frame; None for people off the court."""
    xy = tuple(map(float, court.to_court(H, p.foot)[0]))
    return (court.team_of(xy, us_side) if court.on_court(xy, 1.5) else None), xy


def _ensure_frames(stem: str, start: int, end: int) -> None:
    """Cache the rally's frames as JPEGs so stepping and playback are instant."""
    folder = PROCESSED / stem / "frames"
    folder.mkdir(exist_ok=True)
    with _extract_lock:
        missing = [f for f in range(start, end) if not (folder / f"{f}.jpg").exists()]
        if not missing:
            return
        cap = cv2.VideoCapture(str(_video(stem)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, missing[0])
        for frame in range(missing[0], missing[-1] + 1):
            ok, img = cap.read()
            if not ok:
                break
            target = folder / f"{frame}.jpg"
            if not target.exists():
                cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cap.release()


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


@app.get("/api/videos")
def videos() -> list[dict]:
    out = []
    for path in sorted(RAW.glob("*.mp4")):
        record = PROCESSED / path.stem / "match.json"
        if record.exists():
            match = MatchRecord.model_validate_json(record.read_text())
            if match.rallies:
                done = sum(r.reviewed or r.valid is False for r in match.rallies)
                out.append({"stem": path.stem, "rallies": len(match.rallies), "done": done})
    return out


@app.get("/api/{stem}/rallies")
def rallies(stem: str) -> list[dict]:
    return [
        {
            "index": r.index,
            "set": r.set_number,
            "start": r.start_frame,
            "end": r.end_frame,
            "state": "rejected" if r.valid is False else "done" if r.reviewed else "open",
        }
        for r in _load(stem).rallies
    ]


@app.get("/api/{stem}/rally/{index}")
def rally(stem: str, index: int) -> dict:
    match = _load(stem)
    r = match.rallies[index]
    _ensure_frames(stem, r.start_frame, r.end_frame)
    H = np.array(match.homography)
    us_side = _us_side(match, r)
    boxes = {}
    for frame, obs in _players(str(players_file(PROCESSED / stem, r))).items():
        boxes[frame] = [
            {"track": p.track_id, "box": [round(v) for v in p.box], "team": _team_of(p, H, us_side)[0]}
            for p in obs
        ]
    xs, ys = _ball(stem, match.video.n_frames, match.video.width)
    ball = {
        f: [round(float(xs[f])), round(float(ys[f]))]
        for f in range(r.start_frame, r.end_frame)
        if not np.isnan(xs[f])
    }
    return {
        "rally": r.model_dump(mode="json"),
        "fps": match.video.fps,
        "size": [match.video.width, match.video.height],
        "us_side": us_side,
        "boxes": boxes,
        "ball": ball,
    }


@app.get("/frame/{stem}/{frame}.jpg")
def frame_image(stem: str, frame: int) -> FileResponse:
    path = PROCESSED / stem / "frames" / f"{frame}.jpg"
    if not path.exists():
        _ensure_frames(stem, frame, frame + 1)
    return FileResponse(path, headers={"Cache-Control": "max-age=3600"})


@app.post("/api/{stem}/rally/{index}/edit")
def edit(stem: str, index: int, e: Edit) -> dict:
    with _save_lock:
        match = _load(stem)
        r = match.rallies[index]
        contact = next((c for c in r.contacts if c.frame == e.frame), None)
        team = Team(e.value) if e.value in ("us", "them") else None

        if e.op == "serve" and team:
            _log(stem, r, e, r.serving_team, team)
            r.serving_team, r.serving_reviewed, r.valid = team, True, True
        elif e.op == "not_rally":
            _log(stem, r, e, r.valid, False)
            r.valid = False
        elif e.op == "winner":
            _log(stem, r, e, r.winner, team)
            r.winner, r.winner_reviewed = team, team is not None
        elif e.op == "done":
            # Whatever was left unedited in a finished rally counts as confirmed,
            # except contacts nobody was attached to: those were never real touches.
            for c in r.contacts:
                c.reviewed = True
                c.deleted = c.deleted or c.team is None
            _log(stem, r, e, False, True)
            r.reviewed, r.valid = True, True
        elif e.op == "reopen":
            _log(stem, r, e, r.reviewed, False)
            r.reviewed, r.valid = False, None
        elif e.op == "add" and e.frame is not None:
            if contact is not None:
                raise HTTPException(409, "there is already a touch at this frame")
            xs, ys = _ball(stem, match.video.n_frames, match.video.width)
            x, y = ball_at(xs, ys, e.frame)
            contact = Contact(frame=e.frame, x=x, y=y, manual=True)
            _assign(match, r, contact, e.track_id, stem)
            r.contacts = sorted(r.contacts + [contact], key=lambda c: c.frame)
            relabel(r, match.video.fps)  # propose an action before locking the touch in
            contact.reviewed = True
            _log(stem, r, e, None, {"track": e.track_id, "action": contact.action})
        elif contact is None:
            raise HTTPException(404, f"no touch at frame {e.frame}")
        elif e.op == "confirm":
            _log(stem, r, e, None, {"track": contact.track_id, "action": contact.action})
            contact.reviewed, contact.deleted = True, False
        elif e.op == "delete":
            _log(stem, r, e, {"track": contact.track_id, "action": contact.action}, None)
            contact.reviewed = contact.deleted = True
        elif e.op == "action":
            _log(stem, r, e, contact.action, e.value)
            contact.action, contact.reviewed, contact.deleted = Action(e.value), True, False
        elif e.op == "player":
            _log(stem, r, e, contact.track_id, e.track_id)
            _assign(match, r, contact, e.track_id, stem)
            contact.reviewed, contact.deleted = True, False
        else:
            raise HTTPException(400, f"unknown edit {e.op}")

        relabel(r, match.video.fps)
        _save(stem, match)
        return r.model_dump(mode="json")


def _assign(match: MatchRecord, r: Rally, contact: Contact, track_id: int | None, stem: str) -> None:
    """Point a touch at the player with this track id in the touch's frame."""
    obs = _players(str(players_file(PROCESSED / stem, r))).get(contact.frame, [])
    who = next((p for p in obs if p.track_id == track_id), None)
    if who is None:
        raise HTTPException(404, f"no player {track_id} in frame {contact.frame}")
    team, xy = _team_of(who, np.array(match.homography), _us_side(match, r))
    if team is None:
        raise HTTPException(400, "that person is not on the court")
    contact.track_id, contact.team, contact.court_xy, contact.ambiguous = who.track_id, team, xy, False


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
