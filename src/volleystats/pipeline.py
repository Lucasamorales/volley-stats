"""Run the stages on rallies of one video.

Usage: python -m volleystats.pipeline VIDEO all
       python -m volleystats.pipeline VIDEO RALLY [RALLY ...]

`all` analyses every rally inside a set and saves the result to match.json,
keeping anything a person has already reviewed. Naming rallies instead (an
index, or fN for the rally containing frame N) prints their touches and renders
an annotated clip without saving.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import court
from .actions import label_actions
from .attribution import nearest_player
from .contacts import ball_at, detect_contacts
from .rallies import segment_rallies
from .schema import Contact, PlayerObs, Rally, Side, Team
from .selection import load_track

AMBIGUOUS_MARGIN = 0.15  # box heights; below this the toucher is a guess
END_LINE_ZONE = 0.5  # metres inside an end line that still count as a serving position
SAME_TOUCH = 3  # frames; a re-detected contact this close to a reviewed one is the same touch
GROWTH = 15  # frames a fresh span must add to a finished rally before it is reopened


def track_sides(
    players: dict[int, list[PlayerObs]], H: np.ndarray, margin: float = 1.5
) -> dict[int, Side | None]:
    """Court half each track spends most of its time in; None for people off the court."""
    votes: dict[int, Counter] = {}
    for obs in players.values():
        for p in obs:
            xy = tuple(court.to_court(H, p.foot)[0])
            side = court.side_of(xy) if court.on_court(xy, margin) else None
            votes.setdefault(p.track_id, Counter())[side] += 1
    return {track: c.most_common(1)[0][0] for track, c in votes.items()}


def relabel(rally: Rally, fps: float) -> None:
    """Recompute rule labels and the serving team from the rally's current touches. In place.

    Unattributed and rejected contacts are left out so they cannot break the
    touch order; labels a person has set are kept by `label_actions`.
    """
    touches = rally.touches
    if not touches:
        return
    first = touches[0]
    if rally.serving_reviewed:
        serve_seen = first.team is rally.serving_team
    elif first.court_xy is not None:
        # A serve is only seen when the first toucher stands behind an end line; a
        # server on the near side is out of frame, so the first touch is the receive.
        y = first.court_xy[1]
        serve_seen = y < END_LINE_ZONE or y > court.LENGTH - END_LINE_ZONE
    else:
        serve_seen = False
    label_actions(touches, fps, serve_seen=serve_seen)
    if not rally.serving_reviewed:
        rally.serving_team = first.team if serve_seen else first.team.other


def analyze_rally(
    index: int,
    start: int,
    end: int,
    xs: np.ndarray,
    ys: np.ndarray,
    players: dict[int, list[PlayerObs]],
    H: np.ndarray,
    us_side: Side,
    fps: float,
    roster: dict[int, str] | None = None,
    set_number: int = 1,
) -> Rally:
    """Contacts in frames [start, end), each attributed to a player and given a rule label."""
    sides = track_sides({f: p for f, p in players.items() if start <= f < end}, H)
    contacts = []
    for offset, score in detect_contacts(xs[start:end], ys[start:end]):
        frame = start + offset
        ball = ball_at(xs, ys, frame)
        on_court = [p for p in players.get(frame, []) if sides.get(p.track_id) is not None]
        who, margin = nearest_player(ball, on_court)
        c = Contact(frame=frame, x=ball[0], y=ball[1], score=score)
        if who is not None:
            c.track_id = who.track_id
            c.team = Team.US if sides[who.track_id] is us_side else Team.THEM
            c.court_xy = tuple(map(float, court.to_court(H, who.foot)[0]))
            c.player = (roster or {}).get(who.track_id)
            c.ambiguous = margin < AMBIGUOUS_MARGIN
        contacts.append(c)
    rally = Rally(index=index, set_number=set_number, start_frame=start, end_frame=end, contacts=contacts)
    relabel(rally, fps)
    return rally


def merge_rallies(old: list[Rally], new: list[Rally], fps: float) -> list[Rally]:
    """Combine a fresh analysis with saved rallies without losing a person's work.

    A rally a person finished (or rejected) is kept exactly as saved, and any
    fresh rally overlapping it is dropped. In the other rallies, individually
    reviewed contacts and a reviewed serving team or winner are carried over.
    """

    def overlaps(a: Rally, b: Rally) -> bool:
        return a.start_frame < b.end_frame and b.start_frame < a.end_frame

    def outgrown(o: Rally) -> bool:
        """A fresh span reaches well beyond this rally: it was cut short when reviewed."""
        return any(
            overlaps(o, r) and (o.start_frame - r.start_frame > GROWTH or r.end_frame - o.end_frame > GROWTH)
            for r in new
        )

    # A finished rally that turns out to have been cut short goes back to open:
    # its reviewed touches are carried over and the added frames get fresh proposals.
    frozen = [r for r in old if (r.reviewed or r.valid is False) and not outgrown(r)]
    open_old = [r for r in old if r not in frozen]
    merged = list(frozen)
    for rally in new:
        if any(overlaps(rally, f) for f in frozen):
            continue
        before = [o for o in open_old if overlaps(o, rally)]
        kept = [
            c for o in before for c in o.contacts
            if c.reviewed and rally.start_frame <= c.frame < rally.end_frame
        ]
        fresh = [c for c in rally.contacts if all(abs(c.frame - k.frame) > SAME_TOUCH for k in kept)]
        rally.contacts = sorted(kept + fresh, key=lambda c: c.frame)
        # Several old fragments can fall into one fresh rally: the serve belongs
        # to the earliest, the outcome to the latest.
        for o in sorted(before, key=lambda o: o.start_frame, reverse=True):
            if o.serving_reviewed:
                rally.serving_team, rally.serving_reviewed = o.serving_team, True
        for o in sorted(before, key=lambda o: o.start_frame):
            if o.winner_reviewed:
                rally.winner, rally.winner_reviewed = o.winner, True
        if kept or any(o.serving_reviewed for o in before):
            relabel(rally, fps)
        merged.append(rally)
    merged.sort(key=lambda r: r.start_frame)
    for i, rally in enumerate(merged):
        rally.index = i
    return merged


def players_file(out: Path, rally: Rally) -> Path:
    return out / f"players_{rally.start_frame}_{rally.end_frame}.npz"


if __name__ == "__main__":
    from .players import load_players, track_players
    from .render import render_clip
    from .store import load_match, save_match, video_dir

    video = Path(sys.argv[1])
    match = load_match(video)
    out = video_dir(video)
    H = np.array(match.homography)
    if not match.sets:
        sys.exit("match.json has no sets: record which half our team plays on first")

    def set_at(frame: int):
        """The set being played at this frame, or None during a break."""
        for s in match.sets:
            if s.start_frame <= frame < (s.end_frame or match.video.n_frames):
                return s
        return None

    xs, ys = load_track(out / "ball.csv", match.video.n_frames, width=match.video.width)
    spans = segment_rallies(xs, ys, match.video.fps)
    print(f"{len(spans)} rallies, ball visible in {np.mean(~np.isnan(xs)):.0%} of frames")

    def analyze(index: int) -> tuple[Rally, dict] | None:
        start, end = spans[index]
        current = set_at(start)
        if current is None:
            return None
        tracks = out / f"players_{start}_{end}.npz"
        if not tracks.exists():
            track_players(video, tracks, start, end)
        players = load_players(tracks)
        rally = analyze_rally(
            index, start, end, xs, ys, players, H, current.us_side, match.video.fps,
            set_number=current.set_number,
        )
        return rally, players

    if sys.argv[2:] == ["all"]:
        fresh = []
        for index in range(len(spans)):
            result = analyze(index)
            if result is not None:
                fresh.append(result[0])
            print(f"rally {index + 1}/{len(spans)}", flush=True)
        match = load_match(video)  # pick up reviews saved while this was running
        match.rallies = merge_rallies(match.rallies, fresh, match.video.fps)
        print(f"saved {len(match.rallies)} rallies to", save_match(video, match))
        sys.exit()

    for arg in sys.argv[2:]:
        if arg.startswith("f"):  # f11900 = the rally containing frame 11900
            index = next(i for i, (s, e) in enumerate(spans) if s <= int(arg[1:]) <= e)
        else:
            index = int(arg)
        result = analyze(index)
        if result is None:
            print(f"rally {index} (frame {spans[index][0]}) is outside every set, skipping")
            continue
        rally, players = result
        notes = {}
        for c in rally.contacts:
            flag = "?" if c.ambiguous else ""
            who = f"track {c.track_id}{flag}" if c.track_id is not None else "nobody"
            notes[c.frame] = f"{c.action.value} by {who} ({c.team.value if c.team else '-'})"
            print(f"  frame {c.frame}: {notes[c.frame]}  jump {c.score:.1f}")
        clip = out / "clips" / f"rally_{index:03d}.mp4"
        render_clip(video, clip, rally.start_frame, rally.end_frame, players, xs, ys, H, notes=notes)
        print("wrote", clip)
