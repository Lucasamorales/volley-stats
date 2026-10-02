"""Decide which tracked player touched the ball at a contact."""

import math

from .schema import PlayerObs


def _distance(ball: tuple[float, float], player: PlayerObs) -> float:
    """Ball-to-player distance in units of the player's box height.

    Wrists are used when pose is available; otherwise the distance to the box
    (zero inside it). Normalising by box height keeps the far court comparable
    to the near court.
    """
    bx, by = ball
    x1, y1, x2, y2 = player.box
    height = max(y2 - y1, 1.0)
    if player.wrists:
        d = min(math.hypot(bx - wx, by - wy) for wx, wy in player.wrists)
    else:
        d = math.hypot(max(x1 - bx, 0.0, bx - x2), max(y1 - by, 0.0, by - y2))
    return d / height


def nearest_player(
    ball: tuple[float, float],
    players: list[PlayerObs],
    max_distance: float = 0.6,
) -> tuple[PlayerObs | None, float]:
    """Return the closest player and the confidence margin over the runner-up.

    The margin is the gap between the two nearest players in box heights; a
    small margin means the touch is ambiguous and should go to review. Returns
    (None, 0.0) when nobody is within `max_distance` box heights of the ball.
    """
    ranked = sorted((_distance(ball, p), i) for i, p in enumerate(players))
    if not ranked or ranked[0][0] > max_distance:
        return None, 0.0
    margin = ranked[1][0] - ranked[0][0] if len(ranked) > 1 else max_distance
    return players[ranked[0][1]], margin
