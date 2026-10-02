import numpy as np
import pytest

from volleystats import court
from volleystats.actions import label_actions
from volleystats.attribution import nearest_player
from volleystats.contacts import ball_at, detect_contacts
from volleystats.rallies import segment_rallies
from volleystats.selection import select_track
from volleystats.schema import Action, Contact, PlayerObs, Rally, Side, Team
from volleystats.scoring import check_sets, resolve_winners, running_score
from volleystats.stats import player_table, rally_events

US, THEM = Team.US, Team.THEM


def flight(segments, gravity=0.3, noise=0.5, seed=0):
    """Ball track made of ballistic arcs; each segment is (n_frames, vx, vy)."""
    rng = np.random.default_rng(seed)
    x, y = 400.0, 600.0
    xs, ys = [], []
    for n, vx, vy in segments:
        for _ in range(n):
            xs.append(x)
            ys.append(y)
            x += vx
            y += vy
            vy += gravity
    xs, ys = np.array(xs), np.array(ys)
    return xs + rng.normal(0, noise, len(xs)), ys + rng.normal(0, noise, len(ys))


# --- court ---------------------------------------------------------------


def test_homography_recovers_court_points():
    H_true = np.array([[0.02, 0.001, -3.0], [0.0005, -0.05, 40.0], [0.00001, -0.0006, 1.0]])
    court_pts = np.array(list(court.LANDMARKS.values()))
    homog = np.c_[court_pts, np.ones(len(court_pts))] @ np.linalg.inv(H_true).T
    image_pts = homog[:, :2] / homog[:, 2:3]

    H = court.fit_homography(image_pts[:4], court_pts[:4])
    assert np.allclose(court.to_court(H, image_pts), court_pts, atol=1e-6)


def test_side_and_team():
    assert court.side_of((4, 3)) is Side.NEAR
    assert court.team_of((4, 12), us_side=Side.FAR) is US
    assert court.team_of((4, 12), us_side=Side.NEAR) is THEM
    assert court.on_court((-1, 19)) and not court.on_court((-5, 5))


# --- rallies -------------------------------------------------------------


def test_rallies_split_on_dead_time_and_ignore_static_ball():
    fps = 30
    n = 30 * fps
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    t = np.arange(n)
    for lo, hi in [(60, 300), (480, 700)]:  # two rallies
        xs[lo:hi] = 500 + 200 * np.sin(t[lo:hi] / 10)
        ys[lo:hi] = 300 + 100 * np.cos(t[lo:hi] / 7)
    xs[750:780], ys[750:780] = 100 + 5 * t[750:780], 200  # brief stray ball: too short
    xs[800:880], ys[800:880] = 640, 900  # ball reappears elsewhere, lying on the floor

    spans = segment_rallies(xs, ys, fps)
    assert len(spans) == 2
    (s0, e0), (s1, e1) = spans
    assert s0 <= 61 and 299 <= e0 < 480
    assert e0 < s1 <= 481 and 699 <= e1 < 750


def test_rallies_empty_track():
    assert segment_rallies(np.full(100, np.nan), np.full(100, np.nan), 30) == []


# --- contacts ------------------------------------------------------------


def test_contacts_found_at_velocity_breaks():
    xs, ys = flight([(30, 8, -12), (35, -6, -10), (35, 7, -8)])
    frames = [f for f, _ in detect_contacts(xs, ys)]
    assert len(frames) == 2
    assert abs(frames[0] - 30) <= 2 and abs(frames[1] - 65) <= 2


def test_no_contact_in_free_flight():
    xs, ys = flight([(90, 6, -14)])
    assert detect_contacts(xs, ys) == []


def test_contacts_tolerate_missing_detections():
    xs, ys = flight([(30, 8, -12), (40, -6, -10)])
    xs[[10, 11, 28, 33, 50]] = np.nan
    frames = [f for f, _ in detect_contacts(xs, ys)]
    assert len(frames) == 1 and abs(frames[0] - 30) <= 2


def test_contact_hidden_in_a_detection_gap():
    xs, ys = flight([(40, 8, -12), (40, -6, -10)])
    truth = (xs[40], ys[40])
    xs[34:47] = ys[34:47] = np.nan  # ball lost for 13 frames around the touch
    contacts = detect_contacts(xs, ys)
    assert len(contacts) == 1 and abs(contacts[0][0] - 40) <= 3
    assert np.hypot(*np.subtract(ball_at(xs, ys, contacts[0][0]), truth)) < 40


# --- ball candidate selection ---------------------------------------------


def test_select_track_rejects_distractors_and_keeps_weak_points_on_the_path():
    cands = {f: [(100.0 + 10 * f, 300.0 - 5 * f, 0.9)] for f in range(20)}
    cands[8] = [(180.0, 260.0, 0.3)]  # weak but where the ball should be
    cands[12] = [(900.0, 600.0, 0.95)]  # strong but impossibly far away
    cands[30] = [(500.0, 500.0, 0.3)]  # weak with no recent track
    xs, ys = select_track(cands, 40)
    assert xs[8] == 180.0
    assert np.isnan(xs[12]) and np.isnan(xs[30])
    assert not np.isnan(xs[13])


# --- attribution ---------------------------------------------------------


def test_nearest_player_prefers_wrist_and_respects_range():
    a = PlayerObs(frame=0, track_id=1, box=(100, 100, 160, 300), wrists=[(165, 110)])
    b = PlayerObs(frame=0, track_id=2, box=(200, 100, 260, 300), wrists=[(205, 250)])
    who, margin = nearest_player((170, 105), [a, b])
    assert who.track_id == 1 and margin > 0

    assert nearest_player((900, 50), [a, b]) == (None, 0.0)


def test_nearest_player_without_pose_uses_box():
    a = PlayerObs(frame=0, track_id=1, box=(100, 100, 160, 300))
    b = PlayerObs(frame=0, track_id=2, box=(400, 100, 460, 300))
    assert nearest_player((150, 90), [a, b])[0].track_id == 1


# --- actions -------------------------------------------------------------


def touches(*spec):
    """Contacts from (frame, team, player) or (frame, team, player, court_y)."""
    out = []
    for frame, team, player, *rest in spec:
        xy = (4.5, rest[0]) if rest else None
        out.append(Contact(frame=frame, x=0, y=0, team=team, player=player, court_xy=xy))
    return out


def test_standard_rally_sequence():
    cs = touches(
        (0, THEM, None),
        (40, US, "ana"),
        (70, US, "bea"),
        (100, US, "caro"),
        (140, THEM, None),
        (170, THEM, None),
        (200, THEM, None),
    )
    label_actions(cs, fps=30)
    assert [c.action for c in cs] == [
        Action.SERVE,
        Action.RECEIVE,
        Action.SET,
        Action.ATTACK,
        Action.DIG,
        Action.SET,
        Action.ATTACK,
    ]


def test_block_does_not_use_a_team_touch():
    cs = touches(
        (0, US, "ana"),
        (40, THEM, None),
        (70, THEM, None),
        (100, THEM, None, 9.5),
        (108, US, "bea", 8.6),  # block at the net right after their attack
        (130, US, "caro", 4.0),
        (160, US, "ana", 7.0),
        (190, US, "bea", 8.0),
    )
    label_actions(cs, fps=30)
    assert [c.action for c in cs[4:]] == [Action.BLOCK, Action.DIG, Action.SET, Action.ATTACK]


def test_quick_touch_far_from_net_is_a_dig_not_a_block():
    cs = touches((0, US, "ana"), (40, THEM, None), (70, THEM, None), (78, US, "bea", 2.0))
    label_actions(cs, fps=30)
    assert cs[2].action is Action.ATTACK  # second touch sent over
    assert cs[3].action is Action.DIG


def test_unseen_serve_makes_the_first_touch_a_receive():
    cs = touches((40, US, "ana"), (70, US, "bea"), (100, US, "caro"), (140, THEM, None))
    label_actions(cs, fps=30, serve_seen=False)
    assert [c.action for c in cs] == [Action.RECEIVE, Action.SET, Action.ATTACK, Action.DIG]


def test_reviewed_label_is_kept():
    cs = touches((0, THEM, None), (40, US, "ana"), (70, US, "bea"), (100, THEM, None))
    cs[2].action, cs[2].reviewed = Action.SET, True
    label_actions(cs, fps=30)
    assert cs[2].action is Action.SET


# --- scoring -------------------------------------------------------------


def rally(i, serving, set_number=1, winner=None, contacts=()):
    return Rally(
        index=i,
        set_number=set_number,
        start_frame=i * 100,
        end_frame=i * 100 + 90,
        serving_team=serving,
        winner=winner,
        contacts=list(contacts),
    )


def test_next_server_rule_and_set_boundaries():
    rallies = [
        rally(0, US),
        rally(1, US),
        rally(2, THEM),  # last of set 1: unknown until entered
        rally(3, THEM, set_number=2),
        rally(4, US, set_number=2, winner=US),
    ]
    resolve_winners(rallies)
    assert [r.winner for r in rallies] == [US, THEM, None, US, US]
    assert running_score(rallies) == [(1, 0), (1, 1), (1, 1), (1, 0), (2, 0)]
    assert any("no winner for rallies [2]" in w for w in check_sets(rallies))


def test_manual_winner_is_not_overwritten():
    rallies = [rally(0, US, winner=THEM), rally(1, US)]  # e.g. a replayed point
    resolve_winners(rallies)
    assert rallies[0].winner is THEM


def test_complete_set_has_no_warnings():
    rallies = [rally(i, US) for i in range(25)]
    rallies[-1].winner = US
    resolve_winners(rallies)
    assert running_score(rallies)[-1] == (25, 0)
    assert check_sets(rallies) == []


# --- stats ---------------------------------------------------------------


def labelled(spec):
    cs = touches(*spec)
    label_actions(cs, fps=30)
    return cs


@pytest.mark.parametrize(
    "spec, winner, expected",
    [
        # serve lands untouched
        ([(0, US, "ana")], US, {("ana", "ace")}),
        # serve into the net
        ([(0, US, "ana")], THEM, {("ana", "serve_error")}),
        # they shank our serve
        ([(0, US, "ana"), (40, THEM, None)], US, {("ana", "ace"), (None, "reception_error")}),
        # our attack lands
        (
            [(0, THEM, None), (40, US, "ana"), (70, US, "bea"), (100, US, "caro")],
            US,
            {("caro", "kill")},
        ),
        # our attack goes out
        (
            [(0, THEM, None), (40, US, "ana"), (70, US, "bea"), (100, US, "caro")],
            THEM,
            {("caro", "attack_error")},
        ),
        # they dig our attack but cannot keep it up
        (
            [(0, THEM, None), (40, US, "ana"), (70, US, "bea"), (100, US, "caro"), (140, THEM, None)],
            US,
            {("caro", "kill")},
        ),
    ],
)
def test_terminal_outcomes(spec, winner, expected):
    r = rally(0, spec[0][1], winner=winner, contacts=labelled(spec))
    outcomes = {
        (e.player, e.kind) for e in rally_events(r) if e.kind not in set(Action)
    }
    assert outcomes == expected


def test_stuff_block_scores_for_the_blocker():
    spec = [
        (0, US, "ana"),
        (40, THEM, None),
        (70, THEM, None),
        (100, THEM, None, 9.5),
        (108, US, "bea", 8.6),
    ]
    r = rally(0, US, winner=US, contacts=labelled(spec))
    assert ("bea", "block_point") in {(e.player, e.kind) for e in rally_events(r)}


def test_player_table_counts_points_per_set():
    kill = [(0, THEM, None), (40, US, "ana"), (70, US, "bea"), (100, US, "caro")]
    rallies = [
        rally(0, THEM, winner=US, contacts=labelled(kill)),
        rally(1, US, winner=US, contacts=labelled([(0, US, "caro")])),
        rally(2, THEM, set_number=2, winner=THEM, contacts=labelled(kill)),
    ]
    rows = {(r["player"], r["set"]): r for r in player_table(rallies)}
    assert rows[("caro", 1)]["points"] == 2  # one kill, one ace
    assert rows[("caro", 1)]["attack"] == 1 and rows[("caro", 1)]["serve"] == 1
    assert rows[("caro", 2)]["points"] == 0 and rows[("caro", 2)]["errors"] == 1
    assert rows[("ana", 1)]["receive"] == 1 and rows[("ana", 1)]["points"] == 0
