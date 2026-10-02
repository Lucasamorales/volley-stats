# Volleyball stats from amateur video: state-of-the-art research

Researched 2026-10-01. Figures are as reported by the linked sources; almost all were measured on broadcast or professional footage, not phone/tripod amateur video, so treat them as upper bounds.

## Bottom line

- **Fully automatic per-player stats are not achievable today**, even commercially. The realistic target is a semi-automatic pipeline: models propose every touch, a person confirms or corrects about 10–25% of them per match (estimate, not a measured figure).
- **Player identity is the bottleneck, not action recognition.** A September 2026 study of 66 amateur matches found player-identity accuracy of only 0.34–0.61 on held-out matches, and 0.46–0.59 across nine commercial readers ([arXiv 2609.28049](https://arxiv.org/html/2609.28049v1), vendor-authored preprint, no code or data released).
- **Build around ball contacts.** Open projects and that study converge on the same design: track the ball, find contacts where the trajectory kinks, assign each contact to the nearest player, then classify it with a small model plus volleyball rules.
- **Point winners come from the rules, not the pixels.** "The team that serves next won the rally" was correct in 0.9996 of 8,051 transitions in the same study. Ball landing in/out from one camera is unreliable.
- **No public amateur volleyball dataset exists.** You will need to label some of your own footage; the labelling effort is the real cost of the project.

## Recommended architecture

```
raw video
  │
  ├─ 1. Court calibration      click 4–6 court points once per video → homography
  ├─ 2. Ball tracking          VballNet or WASB volleyball weights, over the whole video (cheap pass)
  ├─ 3. Rally segmentation     ball-in-play state machine → rally clips (skip dead time)
  │
  │  per rally (heavy pass):
  ├─ 4. Player detection       YOLO26 (fine-tuned on your frames)
  ├─ 5. Player tracking        BoT-SORT + ReID; filter to on-court players via homography
  ├─ 6. Team assignment        court side relative to the net (flips each set)
  ├─ 7. Player identity        tag each teammate once per video + appearance gallery
  ├─ 8. Pose                   wrist/arm keypoints per tracked player
  ├─ 9. Contact detection      ball trajectory direction changes → contact frames
  ├─ 10. Attribution           contact → nearest wrist/box → player identity
  ├─ 11. Action type           rules (touch order, court zone, ball height) + small pose classifier
  └─ 12. Rally outcome         next-server rule → point won/lost; last touch → kill/error/ace/block
  │
  ├─ 13. Review UI             confirm low-confidence touches, identities, set boundaries
  └─ 14. Stats store + dashboard   per player, per set, per match
```

Steps 7, 10 and 12 are where errors concentrate, so the review step is part of the design, not an afterthought.

"Points per player" should follow the standard volleyball definition: kills + blocks + aces, with errors counted separately.

## 1. Player detection

| Candidate | Why | Weakness here | Licence |
|---|---|---|---|
| **Ultralytics YOLO26** (Jan 2026) | 40.9–57.5 COCO AP, simplest fine-tuning, built-in tracking ([docs](https://docs.ultralytics.com/models/yolo26)) | AGPL, which only matters if you distribute the project | AGPL-3.0 |
| RF-DETR N–L | 48.4–56.5 AP, designed for fine-tuning ([repo](https://github.com/roboflow/rf-detr)) | Slightly more setup | Apache-2.0 |
| RT-DETRv4, D-FINE | Comparable accuracy ([RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4), [D-FINE](https://github.com/Peterande/D-FINE)) | Research repos, less tooling | Apache-2.0 |

**Pick:** YOLO26 s or m, fine-tuned on a few hundred of your own frames. Switch to RF-DETR if you ever want a permissive licence. YOLO27 was announced in September 2026 but has no weights yet.

## 2. Multi-player tracking

HOTA scores from different papers are not comparable (detector and split differ), so the numbers below are only meaningful within one source.

| Candidate | Evidence | Weakness here | Licence |
|---|---|---|---|
| **BoT-SORT with ReID** | SportsMOT 73.8, DanceTrack 57.8 ([Roboflow trackers](https://github.com/roboflow/trackers)); has camera-motion compensation | ID swaps among same-kit players at the net | Apache-2.0 / AGPL via Ultralytics |
| ByteTrack, OC-SORT | SportsMOT 73.0 / 71.7 in the same table | Motion only | Apache-2.0 |
| McByte | Best in that table: SportsMOT 76.5, DanceTrack 67.2 | Needs a mask model, slower | Apache-2.0 |
| CAMELTrack | SportsMOT 80.3 using appearance + pose ([repo](https://github.com/TrackingLaboratory/CAMELTrack)) | Needs ReID and pose models | Apache-2.0 |
| SAM 3 + Deep-EIoU | SportsMOT test 86.8 HOTA ([arXiv 2606.13033](https://arxiv.org/html/2606.13033v1)) | 0.17 s/frame of SAM time on an RTX 5090; no code found | Paper only |

**Pick:** BoT-SORT with ReID as the baseline. Add a mask-based second pass (SAM-family) only on ambiguous segments if net occlusion causes too many swaps. No tracker keeps identities across a whole match of rotations and substitutions; that is what step 6 is for.

## 3. Court calibration

No general, arbitrary-angle volleyball court model was found. The existing keypoint models are indoor, back-view only ([Court-Keypoint-Detection](https://github.com/asigatchov/Court-Keypoint-Detection), no licence stated). With a tripod the camera is static, so clicking 4–6 court points once per video is cheaper and more robust than training a model. The homography then filters out bystanders and separates the two halves.

## 4. Player identity (the design fork)

| Approach | Evidence | What breaks on your footage |
|---|---|---|
| **Jersey-number OCR** (legibility filter → torso crop → PARSeq → per-track vote) | 87.4% on SoccerNet tracklets, 91.4% on hockey images ([arXiv 2405.13896](https://arxiv.org/html/2405.13896)) | Number only visible when the back or front faces the camera; digits are 15–30 px; cross-domain drops (hockey-trained → soccer 83.9%, soccer-trained → hockey 65.8%) mean you will need to fine-tune on your own crops |
| Fine-tuned small VLM for numbers | Fine-tuned 0.77B Florence-2 beats Gemini 3.1 Pro (0.757 vs 0.737 F1, [arXiv 2609.33450](https://arxiv.org/html/2609.33450)) | Zero-shot VLMs hallucinate numbers on illegible crops |
| **Appearance gallery, tagged once per video** | Rugby sevens full-match tracking from 6 short annotated tracklets per player ([arXiv 2204.04049](https://arxiv.org/abs/2204.04049)) | Same-kit teammates look alike; fails across sessions when clothes change; accuracy with a 6–12 player gallery is unverified |
| Trained person ReID (OSNet, CLIP-ReID) | Collapses across domains: OSNet 83.6% → 3.4% mAP ([arXiv 2601.20598](https://arxiv.org/html/2601.20598)) | Not reliable across different gyms |
| Face recognition | State of the art is about 76% rank-1 on 20×16 px faces ([arXiv 2606.32040](https://arxiv.org/html/2606.32040)) | Faces in a wide shot are that size or smaller and often turned away; suggestion only |

**Team vs opponents:** court side relative to the net, flipped when the whole group swaps ends at set breaks. Colour clustering only works with matching kit.

**Pick for your team (casual clothes, no numbers):**

1. At the start of each video, click each teammate once and give them a roster name (about a minute for 6–12 players).
2. Build a per-video appearance gallery from those tracks (SigLIP2 or OSNet embeddings) and assign every other track to the nearest gallery identity, constrained by court side and "one identity per frame".
3. Send low-confidence assignments to the review queue.
4. Link videos through the roster name you typed, not through automatic matching: appearance ReID breaks when clothes change between sessions and faces are too small.

**Risk:** no source measured how accurate a 6–12 player appearance gallery is on footage like yours, so this is the least-proven part of the design. Phase 1 should measure it on one video before anything is built on top.

Casual clothes may help within a video, since teammates in different shirts should be easier to tell apart than teammates in identical kit (my inference, not a sourced finding). The cost is one manual tagging step per video, and re-tagging if someone changes a top mid-session.

**If you later get numbered jerseys:** add the OCR pipeline mapped to a roster, which removes the per-video tagging and gives cross-video identity automatically.

## 5. Ball tracking

| Candidate | Evidence | Weakness here | Licence |
|---|---|---|---|
| **WASB** | Volleyball F1 88.0, ahead of MonoTrack 85.1 and TrackNetV2 83.6; volleyball weights published ([paper](https://ar5iv.labs.arxiv.org/html/2311.05237), [model zoo](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md)) | Trained on 720p footage; expect a drop on phone video | MIT |
| **VballNet** | Volleyball-specific, F1 0.902 at about 150 FPS on CPU ([repo](https://github.com/asigatchov/fast-volleyball-tracking-inference)) | Self-reported; training data unverified | MIT |
| TrackNetV3/V4/V5, TOTNet | Strong on badminton/tennis | No volleyball weights | Mostly MIT |
| Per-frame YOLO ball detector | mAP50 92.3% in-distribution ([VolleyVision](https://github.com/shukkkur/VolleyVision)) | No temporal model; one amateur project saw only 47–84% detection ([volleybot](https://github.com/rextlfung/volleybot)) | Varies |
| SAM 3, point trackers | — | Tiny blurred balls are a failure mode; not suitable | — |

**Pick:** try VballNet and WASB on your footage, keep the better one, and fine-tune on 500–1,000 labelled frames per new gym or angle if recall is poor (estimate).

## 6. Rally segmentation and point outcome

- **Rally cutting:** a ball-in-play state machine over ball detections is the common open approach; warm-up balls and adjacent courts cause false rallies. A prompted video-language model also segments rallies about as well as anything tested in the amateur study, so it is a reasonable cross-check.
- **Point winner:** detect which side serves each rally and apply the next-server rule. A person enters the last point of each set (it has no next serve) and confirms set boundaries.
- **Scoreboard OCR:** generic tools exist ([ScoreSight](https://github.com/royshil/scoresight)) but none is validated on flip scoreboards; use only as a cross-check.
- **Audio:** whistle detection is established in other sports, but nothing was found for volleyball, and casual matches often have no referee.

## 7. Action recognition

| Approach | Evidence | Weakness here |
|---|---|---|
| **Contacts + rules + small classifier** | Used by [Volleyball-Metrics](https://github.com/kairsato/Volleyball-Metrics), [ClipFarm](https://github.com/ClipFarmVB/ClipFarm) and the amateur study, where a ~2M-parameter model beat frontier VLMs at contact spotting | Depends on ball recall; no published accuracy for nearest-wrist attribution |
| Event spotting (T-DEED, STES) | STES 68.44 mTAP on pro volleyball; block is the weakest class at 59.4 ([VNL-STES](https://hoangqnguyen.github.io/stes/)) | Gives no player; single broadcast viewpoint; needs thousands of events |
| Few-shot spotting (UMEG-Net) | Usable from about 100 labelled clips using pose + ball ([arXiv 2511.14186](https://arxiv.org/html/2511.14186v1)) | Not tested on volleyball |
| Per-frame YOLO action boxes | 92.3% mAP50 in-distribution (VolleyVision) | No temporal reasoning; broadcast-biased; weak "defense" class |
| Video-language models | Good for rally segmentation and pre-labelling | Single-pass VLMs find about a quarter of true contacts; unreliable on identity |

**Rules that carry much of the accuracy:** the first contact of a rally is the serve; touches 1/2/3 on a side are pass/set/attack by default; a contact at the net right after an opponent's attack is a block.

**Datasets for pretraining:** [VNL-STES](https://hoangqnguyen.github.io/stes/) (6,137 events, pro broadcast, licence unverified), [Volleyball Dataset](https://github.com/mostafa-saad/deep-activity-rec) (BSD-2, 9 per-player actions), [MultiSports](https://huggingface.co/datasets/MCG-NJU/SportsAction) (CC BY-NC, 12 volleyball classes).

**Labelling estimate (not sourced):** rules alone might reach 60–70% touch-type accuracy. A useful fine-tuned classifier needs roughly 1,500–3,000 labelled contacts, about 3–6 matches, with at least 150–200 examples of rare classes such as blocks.

## 8. Tooling and compute

**Your machine:** RTX 5070 Ti with 16 GB of VRAM, enough to run and fine-tune the detector, tracker, pose and ball models locally. SAM 3 video tracking is reported at 18–28 GB (third-party figures that conflict), so treat it as cloud-only or skip it. Python 3.14 is installed; a 3.11 or 3.12 environment may be needed for some ML libraries. ffmpeg is not on the PATH.

| Need | Pick | Why | Cost / licence |
|---|---|---|---|
| Boxes and tracks | **CVAT Community**, self-hosted in Docker | Track interpolation; best free option ([pricing](https://www.cvat.ai/pricing/cvat-online)) | Free; SAM 2 video tracking is paid-only |
| Touch/event labels | **Custom review UI** (keypress → frame, action, player) | Doubles as the correction step in the pipeline; Label Studio's [`TimelineLabels`](https://labelstud.io/tags/timelinelabels) is the off-the-shelf alternative | Free |
| Reviewing predictions | FiftyOne | Finds label and prediction errors ([docs](https://docs.voxel51.com/user_guide/annotation.html)) | Free |
| Player pre-labels | Grounding DINO locally, or SAM 2.1 small | Text- or click-prompted boxes to correct rather than draw | Apache-2.0 |
| Rally and action pre-labels | Gemini API | Only major API with native video input, up to 1–3 h per file ([docs](https://ai.google.dev/gemini-api/docs/video-understanding)) | Roughly $0.04–0.81 per video hour at 1 fps (derived estimate); 1 fps is too coarse for contact timing |
| Pose | RTMPose, or YOLO26-pose | RTMPose is Apache-2.0 but MMPose's last release was January 2024, so expect install friction on Windows | Apache-2.0 / AGPL |
| Extra GPU if needed | Kaggle (about 30 free GPU hours/week, unverified) or RunPod (RTX 4090 at $0.34/h, [pricing](https://www.runpod.io/pricing)) | Only for SAM 3 or large training runs | Free / pay-as-you-go |

Avoid Roboflow's free Public plan for this: it reportedly makes uploaded data publicly visible (unverified), which is a problem for footage of your teammates.

**Licence decision to make before writing code:** Ultralytics YOLO26 gives detection, pose and tracking in one package and is the easiest path, but it is AGPL-3.0, so the whole project must be AGPL if you ever share it or run it as a service. An all-Apache stack (RF-DETR + Roboflow `trackers` + RTMPose) avoids that at the cost of more setup. For a private team tool, AGPL has no practical effect.

## Reality check: existing products

Balltime (now part of Hudl) and SportsVisio do this commercially from phone video. Neither publishes accuracy, both have users correct misidentified players, and Balltime asks for 1080p30 from behind the end line on a tripod 8–10 ft high ([FAQ](https://academy.balltime.com/getting-started/faqs)). The closest open end-to-end project is [Volleyball-Metrics](https://github.com/kairsato/Volleyball-Metrics) (MIT, early-stage, no benchmarks).

## Phased plan

1. **Phase 1, one video end to end without actions:** court clicks, player detection and tracking, team side, identity tagging, ball tracking, rally cutting. Output: rally clips with named player tracks. This shows how well each pretrained model holds up on your footage.
2. **Phase 2, touches:** contact detection, attribution, rule-based action types, a review UI for corrections. Corrections become training data.
3. **Phase 3, points and stats:** serving-side detection, next-server rule, set boundaries, terminal-action labelling, stats store and teammate dashboard.
4. **Phase 4, scale out:** fine-tune detector, ball tracker and action classifier on accumulated corrections; run across all videos and gyms.

## Recording advice for future matches

Film from behind the end line, centred, elevated (ideally 8–10 ft), on a fixed tripod with no zoom or panning, at 1080p or better ([Balltime](https://recruiting.balltime.com/baltime-resources/getting-quality-recordings), [Hudl](https://www.hudl.com/support/v3/resources/volleyball-recording-tips)). 60 fps with a fast shutter helps ball tracking; that is general practice rather than a sourced recommendation.

## Questions that change the design

Answered: the team plays in casual clothes, so identity is tag-once-per-video.

Still open:

1. **Scoreboard visible?** Gives a cross-check on point outcomes.
2. **How many videos, and tripod or handheld?** Decides whether per-video manual steps (court clicks, tagging) are acceptable and whether camera-motion compensation is needed.
3. **Private tool or something you might share or publish?** Decides between the Ultralytics (AGPL) stack and the all-Apache stack.
