# Architecture

Status: **proposal, awaiting approval**. Stages marked "built" exist in `src/volleystats/` with tests; the rest is planned.

## Constraints this design is built around

- Fixed tripod, one camera, different gyms and angles per video.
- Team plays in casual clothes with no numbers, so identity cannot be read from jerseys.
- Open-source project; footage of teammates stays private.
- One Windows PC with an RTX 5070 Ti (16 GB). PyTorch 2.11 + CUDA 12.8 is installed and verified on it.
- No model is trusted blindly: every stage writes a confidence, and low-confidence items go to a review queue. Corrections are saved and become training data.

## Pipeline

```
data/raw/<video>.mp4
  │
  ├─ 1. ingest          probe fps/size, write VideoMeta                         planned
  ├─ 2. court           user clicks ≥4 landmarks once → homography              built (maths) / planned (click UI)
  ├─ 3. ball            VballNet ONNX over the whole video → ball track         built
  ├─ 4. rallies         live-ball spans → rally list                            built
  │
  │  per rally:
  ├─ 5. players + pose  YOLO26-pose + BoT-SORT in one pass: boxes, wrists,      built
  │                     track ids; on-court tracks kept via the homography
  ├─ 7. teams           foot point → court half → us / them                     built
  ├─ 8. identity        user tags each teammate once; SigLIP2 gallery matches   planned
  │                     the remaining tracks
  ├─ 9. contacts        breaks in ballistic flight → touch frames               built
  ├─ 10. attribution    touch → nearest wrist/box, with ambiguity margin        built
  ├─ 11. actions        touch-order rules → serve/receive/dig/set/attack/block  built (rules) / planned (classifier)
  ├─ 12. scoring        next-server rule → rally winner, running score          built
  │
  ├─ 13. review         web UI: confirm touches, players, set boundaries        planned
  ├─ 14. stats          per player, per set: attempts, points, errors           built
  └─ 15. dashboard      static HTML, shareable with the team                    planned
```

The ball pass runs first over the whole video because it is cheap (the VballNet repo reports about 150 FPS on CPU) and its output decides which parts of the video the expensive player stack has to look at.

## Chosen stack

| Stage | Choice | Licence | Fallback if it underperforms on our footage |
|---|---|---|---|
| Ball | [VballNet](https://github.com/asigatchov/fast-volleyball-tracking-inference) ONNX (weights ship in the repo; 9 grayscale frames at 288×512) | MIT | [WASB](https://github.com/nttcom/WASB-SBDT) volleyball weights (MIT); fine-tune on our frames |
| Players | Ultralytics YOLO26 + BoT-SORT with ReID | AGPL-3.0 | Fine-tune on a few hundred of our frames |
| Pose | YOLO26-pose (same dependency) | AGPL-3.0 | RTMPose |
| Identity | SigLIP2 embeddings, per-video gallery from user tags | Apache-2.0 | More tags per player; manual assignment in review |
| Court | Manual landmark clicks → DLT homography | ours | Keypoint model trained on CC BY 4.0 court data |
| Actions | Touch-order rules, later a small pose-window classifier trained on review corrections | ours | — |
| Review UI and API | FastAPI + a plain HTML/JS page with keyboard shortcuts | ours | — |
| Stats store | One `match.json` per video + SQLite across matches | ours | — |
| Dashboard | Static HTML generated from SQLite (hostable on GitHub Pages; contains stats only, no footage) | ours | — |

Project licence: **AGPL-3.0-or-later**, forced by Ultralytics.

## Data contracts

Everything derived from a video lives in `data/processed/<video-stem>/`:

| File | Written by | Content |
|---|---|---|
| `match.json` | all stages | `MatchRecord`: video meta, homography, roster (track → name), set configs, rallies with contacts |
| `ball.csv` | ball | `frame, x, y, conf`: up to three candidate blobs per frame; one is selected per frame at load time |
| `players_<start>_<end>.npz` | players, pose | arrays `frame, track, box, conf, kpts` (17 COCO keypoints per person) |
| `clips/rally_NNN.mp4` | pipeline | annotated clip per rally for hand-checking |
| `embeddings.npy` | identity | one appearance vector per track |
| `corrections.jsonl` | review | append-only log of every human edit |

The models are defined in [schema.py](../src/volleystats/schema.py). A `Contact` carries frame, ball position, track, player name, team, court position, action and a `reviewed` flag; anything a person has confirmed is never overwritten by a re-run.

## What a person does per video

1. Click 4–6 court landmarks (once; the camera does not move).
2. Click each teammate and pick their name (about a minute).
3. Mark where each set starts and which half the team is on. The pipeline suggests these from long pauses and the group swapping ends.
4. Enter the winner of the last rally of each set (it has no "next serve").
5. Work through the review queue: touches with an ambiguous player, unknown action, or a score that does not add up.

## How points are counted

- Serving team = the team of the rally's first attributed touch. If that touch has no player, the rally goes to review.
- Rally winner = the team that serves the next rally.
- Set length defaults to 25 points for the sanity check; a deciding set to 15 needs the target passed explicitly.
- A player's **points** = kills + aces + block points. Errors (serve, attack, reception, set) are tracked separately.
- If the losing team touches last: a shanked receive credits an ace to the server; a failed dig or a block touch that goes out credits a kill to the attacker.

## Findings from the first real video (720p30, elevated end-line tripod)

- **Court:** the near end line is out of frame; eight landmarks (far corners, both attack lines, centre line) fit with 2–11 cm reprojection error.
- **Players:** YOLO26m-pose + BoT-SORT kept all six near-court players on one track id each through a 10 s test segment, and bench people fall outside the court margin. Far-court players behind the net are detected less reliably.
- **Ball:** VballNet sees the ball in the air against the walls but almost never below the net line against the floor and players (2–4% of detections below y=300 at the default threshold). Near-court touches therefore happen inside detection gaps, and `contacts.py` infers them from where the arcs before and after the gap meet.
- **General "sports ball" detection (YOLO26 COCO) is not a fix:** it locks onto a stationary object and fires on heads and shoes.
- **Rally cutting over-merges:** ball motion alone marks about three quarters of the video as live play.

## Known weak points

- **Identity with a small gallery is unmeasured.** No source reports accuracy for a 6–12 player tagged gallery on footage like this. Phase 1 measures it first.
- **Ball recall on phone footage.** Published figures are from other footage. Gaps shorter than the contact window are tolerated; longer ones lose touches.
- **Rally cutting from ball motion alone** will merge rallies when the ball is rolled straight back to the server. Serve detection (a player behind the end line, ball tossed up) is the planned second cue.
- **Contact thresholds** (`min_change`, `min_gap`) are in pixels per frame and need tuning per resolution and frame rate.
- **Far-court players are small**, so pose and attribution are weaker for the opponents. That mostly matters for telling a block from a dig.

## Reference projects

| Project | Licence | What we take |
|---|---|---|
| [fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference) | MIT | The ball model and its inference recipe |
| [ClipFarm](https://github.com/ClipFarmVB/ClipFarm) | AGPL-3.0-or-later | The idea of finding contacts as deviations from ballistic flight (our implementation is independent) |
| [Volleyball-Metrics](https://github.com/kairsato/Volleyball-Metrics) | MIT | Closest end-to-end design; its CC BY 4.0 court-keypoint and ball datasets are candidates for fine-tuning. Its rally classifier checkpoint is non-commercial, so we do not use it |
| [volleyball_analytics](https://github.com/masouduut94/volleyball_analytics) | GPL-2.0 | Ideas only. GPL-2.0 code cannot be copied into an AGPL-3.0 project |

## Phases

1. **One video, players and ball.** Ingest, court clicks, ball track, rally cutting, player tracks, tagging, identity matching. Done when a rally clip can be rendered with named boxes and the ball, and identity accuracy has been measured on 10 hand-checked rallies.
2. **Touches.** Contacts, attribution and rule labels on real tracks; review UI; corrections log.
3. **Points and stats.** Serving side, set boundaries, scoring, SQLite, dashboard.
4. **Scale.** Fine-tune ball, detector and an action classifier from corrections; batch all videos.
