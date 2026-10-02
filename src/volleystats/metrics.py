"""How well the pipeline's proposals held up against a person's review.

Usage: python -m volleystats.metrics

Only rallies marked done count. Player and action accuracy come from the
corrections log: a kept touch whose player or action was never edited was right.
"""

import json
from collections import Counter

from .schema import MatchRecord
from .store import PROCESSED


def summarise(match: MatchRecord, corrections: list[dict]) -> Counter:
    edited = {(e["rally_start"], e["frame"], e["op"]) for e in corrections}
    n: Counter = Counter()
    for rally in match.rallies:
        if rally.valid is False:
            n["rallies rejected"] += 1
        if not rally.reviewed:
            continue
        n["rallies checked"] += 1
        for c in rally.contacts:
            if c.manual:
                n["missed"] += 1
            elif c.deleted:
                n["false"] += 1
            else:
                n["kept"] += 1
                n["player right"] += (rally.start_frame, c.frame, "player") not in edited
                n["action right"] += (rally.start_frame, c.frame, "action") not in edited
    return n


def report(n: Counter) -> str:
    def rate(a: int, b: int) -> str:
        return f"{a}/{b} ({a / b:.0%})" if b else "n/a"

    kept = n["kept"]
    return "\n".join(
        [
            f"rallies checked: {n['rallies checked']}, rejected as not a rally: {n['rallies rejected']}",
            f"precision (proposed touches that were real): {rate(kept, kept + n['false'])}",
            f"recall (real touches that were proposed):    {rate(kept, kept + n['missed'])}",
            f"right player among kept touches:             {rate(n['player right'], kept)}",
            f"right action among kept touches:             {rate(n['action right'], kept)}",
        ]
    )


if __name__ == "__main__":
    total: Counter = Counter()
    for record in sorted(PROCESSED.glob("*/match.json")):
        match = MatchRecord.model_validate_json(record.read_text())
        log = record.with_name("corrections.jsonl")
        corrections = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        n = summarise(match, corrections)
        if n["rallies checked"] or n["rallies rejected"]:
            print(f"\n{record.parent.name}\n{report(n)}")
            total += n
    print(f"\nall videos\n{report(total)}")
