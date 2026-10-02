"""Run the stages on one rally and render a clip for checking.

Usage: python -m volleystats.pipeline VIDEO RALLY [RALLY ...]

RALLY is a rally index, or fN for the rally containing frame N.
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
    # Unattributed contacts stay in the rally for review but must not break the touch order.
    touches = [c for c in contacts if c.team is not None]
    serving = None
    if touches:
        # A serve is only seen when the first toucher stands behind an end line; a
        # server on the near side is out of frame, so the first touch is the receive.
        y = touches[0].court_xy[1]
        serve_seen = y < END_LINE_ZONE or y > court.LENGTH - END_LINE_ZONE
        label_actions(touches, fps, serve_seen=serve_seen)
        serving = touches[0].team if serve_seen else touches[0].team.other
    return Rally(
        index=index,
        set_number=set_number,
        start_frame=start,
        end_frame=end,
        serving_team=serving,
        contacts=contacts,
    )


if __name__ == "__main__":
    from .players import load_players, track_players
    from .render import render_clip
    from .store import load_match, video_dir

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

    for arg in sys.argv[2:]:
        if arg.startswith("f"):  # f11900 = the rally containing frame 11900
            index = next(i for i, (s, e) in enumerate(spans) if s <= int(arg[1:]) <= e)
        else:
            index = int(arg)
        start, end = spans[index]
        current = set_at(start)
        if current is None:
            print(f"rally {index} (frame {start}) is outside every set, skipping")
            continue
        tracks = out / f"players_{start}_{end}.npz"
        if not tracks.exists():
            track_players(video, tracks, start, end)
        players = load_players(tracks)
        rally = analyze_rally(
            index, start, end, xs, ys, players, H, current.us_side, match.video.fps,
            set_number=current.set_number,
        )
        notes = {}
        for c in rally.contacts:
            flag = "?" if c.ambiguous else ""
            who = f"track {c.track_id}{flag}" if c.track_id is not None else "nobody"
            notes[c.frame] = f"{c.action.value} by {who} ({c.team.value if c.team else '-'})"
            print(f"  frame {c.frame}: {notes[c.frame]}  jump {c.score:.1f}")
        clip = out / "clips" / f"rally_{index:03d}.mp4"
        render_clip(video, clip, start, end, players, xs, ys, H, notes=notes)
        print("wrote", clip)
