# Phase 1F: how precise does video tracking need to be?

Simulated video-tracking errors on the BDB 2026 test plays (weeks 16–18), scoring the 1C ensemble (`main`) and a noise-augmented ensemble (`robust`: trained with random drift up to σ = 1 yd, a few missed detections, and the same smoothing used at inference). Full tables: [main](phase1f_noise_main.json), [robust](phase1f_noise_robust.md).

**Tolerance** = largest error that loses ≤ 3 pts man/zone and ≤ 5 pts shell at the snap (same for both models, each against its own clean score):

| Error type | Tolerance |
|---|---|
| Fixed per-player offset | **1.0 yd** |
| Slow drift (ρ = 0.97 per 0.1 s), smoothed | **0.5 yd** |
| Fast jitter (ρ = 0.8), smoothed | 0.25 yd |
| Fast jitter, unsmoothed | 0 yd: jitter becomes fake velocity, so tracks must be smoothed |

Noise augmentation costs ~1 pt clean but holds up much better under realistic corruption (accuracy at the snap, 7-class / man-zone / shell):

| Condition | main | robust |
|---|---|---|
| Clean | 73.8 / 91.1 / 88.2 | 72.2 / 90.4 / 86.9 |
| 1 yd drift, smoothed | 41.1 / 77.7 / 63.3 | **54.3 / 85.7 / 73.2** |
| 20% missed detections | 53.6 / 80.2 / 68.9 | **56.4 / 84.4 / 73.5** |
| 1 yd drift + 10% missed + ID swaps | 17.3 / 38.0 / 32.2 | **36.5 / 68.8 / 60.8** |
| Identity swaps only (half of plays) | 73.2 / 90.6 / 88.0 | 71.4 / 90.1 / 86.7 |

**Gate:** passes (≥ 0.5 yd) for the error types registration mostly produces (fixed offsets, slow drift). The spec for video is therefore **≤ 0.5 yd of slow error with smoothing**, and missed detections matter more than identity swaps. At ~1 yd, man/zone and post-snap reads survive (robust: 85.7% man/zone at the snap, 85.7% 7-class at +2 s), but exact coverage *at the snap* is the read most at risk. Use the robust models for video.
