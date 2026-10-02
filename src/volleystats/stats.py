"""Turn labelled rallies into per-player, per-set statistics."""

from collections import Counter, defaultdict

from .schema import Action, Contact, Rally, StatEvent, Team

# Terminal outcomes that put a point on the board for the player credited.
POINT_KINDS = ("kill", "ace", "block_point")
ERROR_KINDS = ("serve_error", "attack_error", "reception_error", "set_error")


def rally_events(rally: Rally) -> list[StatEvent]:
    """One event per touch (the attempt), plus the outcome events that ended the rally."""

    def event(c: Contact, kind: str) -> StatEvent:
        return StatEvent(
            set_number=rally.set_number,
            rally_index=rally.index,
            team=c.team,
            player=c.player,
            kind=kind,
        )

    touches = [c for c in rally.contacts if c.team is not None]
    events = [event(c, c.action.value) for c in touches if c.action is not Action.UNKNOWN]
    if rally.winner is None or not touches:
        return events

    last = touches[-1]
    if last.team is rally.winner:
        kind = {Action.SERVE: "ace", Action.BLOCK: "block_point"}.get(last.action, "kill")
        events.append(event(last, kind))
        return events

    # The losing team touched the ball last.
    own_error = {
        Action.SERVE: "serve_error",
        Action.ATTACK: "attack_error",
        Action.SET: "set_error",
        Action.RECEIVE: "reception_error",
    }.get(last.action)
    if own_error:
        events.append(event(last, own_error))
    # A shanked receive is an ace; a failed dig or a block touch that goes out is a kill.
    credit = next((c for c in reversed(touches[:-1]) if c.team is rally.winner), None)
    if credit is not None:
        if last.action is Action.RECEIVE and credit.action is Action.SERVE:
            events.append(event(credit, "ace"))
        elif last.action in (Action.DIG, Action.BLOCK) and credit.action is Action.ATTACK:
            events.append(event(credit, "kill"))
    return events


def player_table(rallies: list[Rally], team: Team = Team.US) -> list[dict]:
    """One row per (player, set) with a count of each event kind, points and errors."""
    counts: dict[tuple[str, int], Counter] = defaultdict(Counter)
    for rally in rallies:
        for e in rally_events(rally):
            if e.team is team and e.player is not None:
                counts[(e.player, e.set_number)][e.kind] += 1

    rows = []
    for (player, set_number), c in sorted(counts.items()):
        rows.append(
            {
                "player": player,
                "set": set_number,
                **dict(sorted(c.items())),
                "points": sum(c[k] for k in POINT_KINDS),
                "errors": sum(c[k] for k in ERROR_KINDS),
            }
        )
    return rows
