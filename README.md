# volley-stats

Per-player volleyball statistics (serves, receives, sets, attacks, blocks, points, errors per set) from amateur tripod video.

The pipeline is semi-automatic by design: models propose every touch and who made it, and a person confirms the uncertain ones. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and [docs/RESEARCH.md](docs/RESEARCH.md) for the research behind it.

## Status

Working on real footage: court calibration, ball tracking (VballNet), player and pose tracking (YOLO26-pose + BoT-SORT), touch detection, touch attribution and rule-based action labels, rendered as an annotated clip per rally.

Rough: rally cutting (merges and splits rallies) and touch detection (about as many false touches as real ones on the first test rally).

Also working: a local review page for confirming and correcting touches, and a metrics script that scores the pipeline against those reviews.

Not built yet: identity tagging, scoring on real rallies and the dashboard.

## Running it on a video

The ball model comes from [fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference) (MIT); clone it into `third_party/vballnet`.

```bash
uv run python -m volleystats.calibrate data/raw/VIDEO.mp4
```

```bash
uv run python -m volleystats.ball data/raw/VIDEO.mp4 third_party/vballnet/models/VballNetV4c_seq9_grayscale_20260908_213829.onnx data/processed/VIDEO/ball.csv
```

```bash
uv run python -m volleystats.pipeline data/raw/VIDEO.mp4 all
```

The last command analyses every rally and saves it to `data/processed/VIDEO/match.json`. Re-running it keeps anything already reviewed. Before it can run, `match.json` needs the sets: the frame range of each and which half the team plays on.

## Reviewing

```bash
uv run python -m volleystats.review
```

Then open http://127.0.0.1:8000. For each rally, say who served, step through the proposed touches (confirm, reject, fix the action or the player, add missed ones) and mark it done. Every edit is saved at once and logged to `corrections.jsonl`. The page caches frames under `data/processed/VIDEO/frames/`; that folder can be deleted at any time.

```bash
uv run python -m volleystats.metrics
```

This prints precision, recall, and player and action accuracy over the rallies marked done.

## Where to put videos

Copy match recordings into `data/raw/`. Nothing under `data/` is ever committed.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, for the model stages, an NVIDIA GPU and ffmpeg (`winget install ffmpeg`).

```bash
uv sync
```

```bash
uv run pytest
```

The model stack is an optional extra:

```bash
uv sync --extra ml
```

## Privacy

Footage, frames, crops and labels show real people. `data/`, `weights/` and all video files are git-ignored; keep it that way, and do not upload teammates' footage to hosted labelling services with public projects.

## Licence

AGPL-3.0-or-later. The model stages depend on Ultralytics YOLO, which is AGPL-3.0, so the project as a whole is too.
