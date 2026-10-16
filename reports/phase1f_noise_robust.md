# Phase 1F: noise tolerance

1C ensemble on BDB 2026 test plays (weeks 16–18) with simulated video-tracking errors. Accuracy at the snap / +2.0 s. Drift = AR(1) position error (ρ = 0.8 per 0.1 s), σ in yards; drop = share of player-frames missing; swap = share of plays with one identity switch.

| Condition | Man/zone snap | +2.0 s | 7-class snap | +2.0 s | Shell snap | +2.0 s |
|---|---|---|---|---|---|---|
| clean | 90.4% | 95.6% | 72.2% | 90.5% | 86.9% | 96.6% |
| clean_smoothed | 90.3% | 95.6% | 71.5% | 90.7% | 85.9% | 96.5% |
| drift_0.25yd | 87.4% | 95.8% | 58.3% | 90.7% | 76.0% | 96.5% |
| drift_0.25yd_smoothed | 89.6% | 95.8% | 68.1% | 90.9% | 84.2% | 96.5% |
| slow_0.25yd_smoothed | 90.2% | 95.6% | 70.5% | 90.5% | 85.5% | 96.3% |
| offset_0.25yd | 90.5% | 95.5% | 72.2% | 90.5% | 86.9% | 96.5% |
| drift_0.5yd | 83.0% | 95.2% | 50.7% | 88.0% | 70.8% | 95.1% |
| drift_0.5yd_smoothed | 88.0% | 95.6% | 62.8% | 90.4% | 79.9% | 96.1% |
| slow_0.5yd_smoothed | 90.3% | 95.4% | 68.4% | 89.8% | 84.8% | 96.1% |
| offset_0.5yd | 90.2% | 95.1% | 71.5% | 90.2% | 86.6% | 96.2% |
| drift_1.0yd | 71.5% | 89.9% | 38.2% | 76.9% | 62.3% | 90.9% |
| drift_1.0yd_smoothed | 85.7% | 93.8% | 54.4% | 85.7% | 73.2% | 94.1% |
| slow_1.0yd_smoothed | 87.8% | 94.9% | 63.5% | 87.0% | 80.5% | 94.5% |
| offset_1.0yd | 89.0% | 94.1% | 67.9% | 88.1% | 85.2% | 95.7% |
| drift_1.5yd | 57.6% | 73.6% | 32.4% | 53.6% | 57.2% | 76.4% |
| drift_1.5yd_smoothed | 81.7% | 91.8% | 49.2% | 78.1% | 70.3% | 90.9% |
| slow_1.5yd_smoothed | 85.0% | 92.7% | 55.1% | 80.8% | 74.1% | 92.2% |
| offset_1.5yd | 87.1% | 92.2% | 63.5% | 83.5% | 82.7% | 93.3% |
| drift_2.0yd | 47.9% | 47.9% | 27.8% | 33.6% | 56.1% | 62.7% |
| drift_2.0yd_smoothed | 78.3% | 86.9% | 43.9% | 69.4% | 66.9% | 86.2% |
| slow_2.0yd_smoothed | 81.8% | 90.1% | 48.4% | 74.6% | 69.6% | 88.7% |
| offset_2.0yd | 85.3% | 90.2% | 59.5% | 79.1% | 80.3% | 91.5% |
| drift_3.0yd | 41.1% | 29.5% | 25.8% | 23.3% | 54.4% | 52.8% |
| drift_3.0yd_smoothed | 69.3% | 68.3% | 36.6% | 46.5% | 60.3% | 70.8% |
| slow_3.0yd_smoothed | 78.4% | 84.2% | 43.0% | 60.3% | 66.0% | 80.0% |
| offset_3.0yd | 82.3% | 87.5% | 51.4% | 66.9% | 73.6% | 83.5% |
| drop_0.05 | 89.6% | 95.4% | 67.8% | 90.5% | 83.6% | 96.2% |
| drop_0.1 | 87.6% | 95.8% | 63.6% | 90.7% | 79.4% | 96.3% |
| drop_0.2 | 84.4% | 94.9% | 56.4% | 89.4% | 73.5% | 96.2% |
| drop_0.3 | 80.2% | 94.5% | 48.6% | 88.2% | 67.3% | 95.8% |
| swap_0.25 | 90.3% | 95.3% | 72.0% | 90.2% | 86.8% | 96.5% |
| swap_0.5 | 90.1% | 95.2% | 71.4% | 90.2% | 86.7% | 96.6% |
| swap_1.0 | 90.4% | 95.0% | 71.8% | 90.0% | 86.5% | 96.5% |
| combo_1yd_drop0.1_swap0.5 | 68.8% | 89.0% | 36.5% | 73.8% | 60.8% | 88.2% |

**Tolerance (largest σ losing ≤ 3 pts man/zone and ≤ 5 pts shell at the snap), model `robust`:** {'fast_drift_raw': 0.0, 'fast_drift_smoothed': 0.25, 'slow_drift_smoothed': 0.5, 'static_offset': 1.0}. Gate: ≥ 0.5 yd → full video scope; < 0.5 yd → scope video to shell and man/zone, or add human QA.
