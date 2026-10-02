# volley-stats

Per-player volleyball statistics from amateur tripod video. Design in `docs/ARCHITECTURE.md`, research in `docs/RESEARCH.md`.

## Standards

- No comments or metadata referencing AI tools anywhere in the codebase or in commit messages.
- Comments explain the logic and the why behind decisions. Do not state the obvious.
- Commit messages start with one of these prefixes:

| Prefix | Usage |
| :--- | :--- |
| `feat:` | New features or functional additions. |
| `fix:` | Bug fixes or error corrections. |
| `chore:` | Maintenance, dependency updates, routine tasks. |
| `wip:` | Work in progress (intermediate saves). |
| `docs:` | Documentation or comment changes only. |
| `devops:` | CI/CD, build scripts, infrastructure. |

- Only commit with `feat`, `fix` or `devops` once the code is verified to work or the user has approved it.

## Project facts

- The team plays in casual clothes with no jersey numbers; videos are filmed from a fixed tripod behind an end line.
- Footage, frames, crops and labels show real people: nothing under `data/` or `weights/` is ever committed.
- The model stack is an optional extra: `uv sync --extra ml`. Tests run without it: `uv run pytest`.
- Per video, `data/processed/<stem>/match.json` holds the court, the sets (frame range and which half the team plays on) and the rallies. Anything a person reviewed must survive a re-run.
