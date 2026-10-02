"""Rule-based action labels from the order of touches in a rally.

These are priors, not ground truth: they give every touch a sensible default
that a pose classifier or a human reviewer can override.
"""

from .court import NET_Y
from .schema import Action, Contact


def label_actions(
    contacts: list[Contact],
    fps: float,
    block_window_s: float = 0.5,
    net_zone_m: float = 1.5,
    serve_seen: bool = True,
) -> list[Contact]:
    """Fill `action` on each contact that has not been reviewed. Edits in place.

    Rules: the first touch of the rally is the serve; a touch at the net right
    after the opponent's attack is a block (and does not count towards the three
    team touches); otherwise touch 1 is a receive (after a serve) or dig, touch 2
    a set, touch 3 an attack. A second touch that sends the ball over is an attack.

    Pass `serve_seen=False` when the serve happened out of view, so the first
    contact given is the receive.
    """
    touch = 0  # team touches so far in the current possession
    for i, c in enumerate(contacts):
        prev = contacts[i - 1] if i else None
        nxt = contacts[i + 1] if i + 1 < len(contacts) else None

        if c.team is None:
            touch = 0
            if not c.reviewed:
                c.action = Action.SERVE if i == 0 else Action.UNKNOWN
            continue
        if prev is None or prev.team != c.team:
            touch = 0

        if c.reviewed:
            action = c.action  # a human label wins, and still drives the touch count
        elif i == 0 and serve_seen:
            action = Action.SERVE
        elif prev is not None and _is_block(c, prev, fps, block_window_s, net_zone_m):
            action = Action.BLOCK
        else:
            goes_over = nxt is None or nxt.team != c.team
            if touch == 0:
                after_serve = prev is None or prev.action is Action.SERVE
                action = Action.RECEIVE if after_serve else Action.DIG
            elif touch == 1 and not goes_over:
                action = Action.SET
            else:
                action = Action.ATTACK

        c.action = action
        if action not in (Action.SERVE, Action.BLOCK):
            touch += 1
    return contacts


def _is_block(
    c: Contact, prev: Contact, fps: float, block_window_s: float, net_zone_m: float
) -> bool:
    if prev.team is None or prev.team == c.team or prev.action is not Action.ATTACK:
        return False
    if (c.frame - prev.frame) > block_window_s * fps:
        return False
    if c.court_xy is not None and abs(c.court_xy[1] - NET_Y) > net_zone_m:
        return False
    return True
