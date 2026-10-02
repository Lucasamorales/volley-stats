"""Point outcomes from the serving order: whoever serves next won the rally."""

from itertools import groupby

from .schema import Rally, Team


def resolve_winners(rallies: list[Rally]) -> list[Rally]:
    """Fill `winner` on rallies where it is not already set. Edits in place.

    Within a set, the winner of a rally is the team serving the next one. The
    last rally of each set has no next serve, so its winner stays None until a
    person enters it.
    """
    for _, group in groupby(rallies, key=lambda r: r.set_number):
        in_set = list(group)
        for rally, following in zip(in_set, in_set[1:]):
            if rally.winner is None:
                rally.winner = following.serving_team
    return rallies


def running_score(rallies: list[Rally]) -> list[tuple[int, int]]:
    """(us, them) after each rally, restarting at each set. Unresolved rallies add nothing."""
    out: list[tuple[int, int]] = []
    us = them = 0
    current = None
    for r in rallies:
        if r.set_number != current:
            us = them = 0
            current = r.set_number
        us += r.winner is Team.US
        them += r.winner is Team.THEM
        out.append((us, them))
    return out


def check_sets(rallies: list[Rally], target: int = 25) -> list[str]:
    """Human-readable warnings where the reconstructed score looks wrong."""
    warnings: list[str] = []
    scores = running_score(rallies)
    for set_number, group in groupby(zip(rallies, scores), key=lambda rs: rs[0].set_number):
        items = list(group)
        unresolved = [r.index for r, _ in items if r.winner is None]
        if unresolved:
            warnings.append(f"set {set_number}: no winner for rallies {unresolved}")
        us, them = items[-1][1]
        if not unresolved and (max(us, them) < target or abs(us - them) < 2):
            warnings.append(
                f"set {set_number}: final score {us}-{them} is not a finished set to {target}"
            )
    return warnings
