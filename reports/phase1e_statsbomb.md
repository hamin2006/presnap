# Phase 1E: StatsBomb transfer (all 22 players, pre-snap motion, blitz)

StatsBomb TB12DB tracking, 2018–22 (5,207 quality-filtered labeled dropbacks; test = 809 plays from held-out games). All 22 players are tracked from 3.0 s before the snap, so there is **no missing-rusher leak**. Labels are NGS coverage via nflverse; blitz = 5+ pass rushers (NGS charting). Full tables: [scratch](phase1c_sb_scratch.md), [warm start](phase1c_sb_ft.md).

| At the snap | 7-class | Man/zone (AUC) | Played shell | Blitz AUC |
|---|---|---|---|---|
| GBM, snap features | 65.1% | 87.6% (0.944) | 80.5% | 0.779 |
| Transformer from scratch | 67.9% | 89.0% (0.959) | 83.6% | **0.812** |
| **Transformer warm-started from the BDB model** | **73.4%** | **90.2% (0.965)** | **85.3%** | 0.797 |

How readable the defense is over time (warm start, 7-class / man / shell / blitz AUC):

| −2.0 s | −1.0 s | snap | +1.0 s | +2.0 s |
|---|---|---|---|---|
| 59 / 86 / 73 / 0.74 | 63 / 88 / 76 / 0.75 | 73 / 90 / 85 / 0.80 | 88 / 96 / 94 / 0.90 | 89 / 97 / 95 / 0.93 |

- **Leak test passed.** With every player tracked, the warm-started transformer beats the GBM by +8.3 pts on 7-class coverage at the snap: the same margin as on BDB, so the BDB gain was not the defender-count leak.
- **Blitz gate passed** (AUC > 0.71 on game-held-out plays): 0.80–0.81 at the snap and 0.75 a full second before it.
- **Pre-snap reads are real but limited:** 2 s before the snap, coverage family is 59% readable (vs 33% for the most common class) and man/zone 86%; most disguise resolves in the first second after the snap.
- **External check:** the warm-started model on 117 NGS-tracked 2020–21 plays it never saw (a third data source) scores 86% man/zone, 65% 7-class and 86% shell at the snap, and 94% / 87% / 97% at +1.0 s.
- Caveat: the offense is always Brady's, and half the plays have NE/TB on defense.
