# Phase 2 status: video → field positions (stopped at the 2C gate)

Data: Kaggle NFL Player Contact Detection, 240 plays of All29 (all-22) video with synced NGS tracking; 117 are dropbacks with NGS coverage labels (via nflverse, 2020–21).

## 2B field registration: NOT validated (correction, 2026-10-16)
Line-based registration (yard lines + sidelines, labeled by the LOS, refined on the painted lines). Fixes made along the way: end lines in end-zone frames, per-frame turf hue (grey-teal turf in Atlanta), top-hat paint detection for low-contrast lines, all sideline pairs (stands railings were taken as the sideline).

| Version | Frames registered | Median error of matched players |
|---|---|---|
| v1 | 47% | 1.04 yd |
| v3 | **69%** | **0.99 yd** (p90 1.42) |

Line-fit residual 1.5 px; no measurable video/tracking sync offset (flat across ±0.3 s).

**Correction:** the "≈1 yd" figure is biased. It matched NGS players to the nearest of ~200 loose detections (sideline people included), so close matches happen even when the registration is wrong. A stricter check (scratch/p2_offset.py, 27 plays): with the fitted registration only **~3 of 22** NGS players land within 0.75 yd of a confident detection; the best per-play field shift scatters widely (dy −3.5 to +2.5 yd, dx up to ±5 yd) with no common bias, and even after the best shift only ~7 of 22 match. A rendered overlay (58187_001691) shows the whole NGS formation ~3 yd off the players. Likely causes: sideline border edge vs centre (±2 yd in y), residual yard-line aliasing, and wide-angle lens distortion that one homography can't model. There is no pixel ground truth for All29, which is what let the biased metric through.

## 2E coach-assisted (simulated clicks): inconclusive
Clicks were simulated by projecting NGS positions through the (unreliable) registration, so they landed off the players and the optical-flow tracks followed grass. The simulation can't be trusted until registration has pixel ground truth. Frame-to-frame registration was also fixed (it slipped exactly one yard line when a middle line was missed during pans; now lines are relabelled from the previous frame).

## 2C player detection: gate failed
Players are ~25–35 px tall (often ~15 px in wide shots) and packed at the line. Share of the 22 NGS players found:

| Detector | Within 2 yd | Within 1 yd |
|---|---|---|
| COCO YOLO11x, tiled + upscaled, **correct registration** | 42% at the snap (56% at −1 s, 28% at +2 s) | 21% |
| COCO YOLO11x, tiled, deployable registration (LOS-anchored), held-out plays | 33–36% within 1.5 yd | – |
| Fine-tuned YOLO11s on NGS-confirmed pseudo-labels | 25% within 1.5 yd (conf 0.02) | – |

1F says the coverage model needs ≲ 0.5 yd error and few missed players (20% missed already costs ~15 pts of 7-class accuracy at the snap). At ~60% missed, the video path can't feed the model, so **2D was not run**. The NGS path on the same 117 plays works: 86% man/zone, 65–68% 7-class, 86–88% shell at the snap.

First fine-tune attempt failed for a fixable reason: labels from projected NGS positions were 20–40 px off ~12 px-wide players (mAP50 0.01). Second attempt used NGS-confirmed detector boxes (mAP50 0.30 on incomplete labels) but inherits the COCO detector's misses, so it can't exceed it.
