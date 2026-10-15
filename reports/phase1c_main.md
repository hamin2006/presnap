# Phase 1C coverage model: main

Transformer (214,476 params, 5-seed ensemble, max_def=None) vs GBM trained at the same moment. Test = weeks 16–18 (2023); 95% game-clustered bootstrap CIs. `agree` = plays where NGS and FTN labels match.

## Man vs zone

| Moment | Transformer | 95% CI | Agree subset | GBM | GBM 95% CI |
|---|---|---|---|---|---|
| snap | **91.1%** | 89.7%–92.3% | 90.4% | 87.2% | 85.8%–88.5% |
| +0.5 s | **93.1%** | 92.0%–94.3% | 93.0% | 88.7% | 87.4%–90.0% |
| +1.0 s | **94.5%** | 93.4%–95.5% | 94.7% | 90.4% | 89.2%–91.8% |
| +1.5 s | **95.1%** | 94.1%–96.0% | 95.4% | 90.9% | 89.6%–92.2% |
| +2.0 s | **95.5%** | 94.6%–96.3% | 96.1% | 91.3% | 89.8%–92.7% |
| throw | **95.9%** | 95.0%–96.8% | 96.8% | 90.5% | 89.4%–91.7% |

## 7-class coverage

| Moment | Transformer | 95% CI | Agree subset | GBM | GBM 95% CI |
|---|---|---|---|---|---|
| snap | **73.8%** | 71.6%–76.1% | 77.4% | 65.7% | 63.4%–68.0% |
| +0.5 s | **82.5%** | 80.5%–84.2% | 86.2% | 72.0% | 69.5%–74.7% |
| +1.0 s | **87.8%** | 86.1%–89.3% | 91.1% | 78.0% | 75.9%–80.3% |
| +1.5 s | **89.8%** | 88.5%–91.0% | 92.5% | 80.7% | 78.8%–82.5% |
| +2.0 s | **90.2%** | 89.0%–91.4% | 92.9% | 81.6% | 79.9%–83.3% |
| throw | **90.6%** | 89.4%–91.7% | 93.4% | 78.6% | 76.9%–80.2% |

## Played shell

| Moment | Transformer | 95% CI | Agree subset | GBM | GBM 95% CI |
|---|---|---|---|---|---|
| snap | **88.2%** | 86.0%–90.0% | 89.8% | 83.3% | 81.0%–85.2% |
| +0.5 s | **92.9%** | 91.5%–94.2% | 94.7% | 88.7% | 86.9%–90.1% |
| +1.0 s | **95.3%** | 94.1%–96.5% | 97.0% | 91.6% | 90.1%–93.1% |
| +1.5 s | **96.0%** | 95.0%–96.9% | 97.2% | 92.5% | 91.4%–93.6% |
| +2.0 s | **95.8%** | 94.9%–96.7% | 97.0% | 92.8% | 91.5%–94.0% |
| throw | **95.9%** | 95.0%–96.7% | 97.1% | 90.9% | 89.3%–92.3% |

## Is the snap advantage just the defender count?

BDB 2026 tracks only defenders who dropped into coverage, so a model could infer how many rushed. Two checks (single seed for the ablation, 5-seed ensemble otherwise; test weeks 16–18):

| Check | 7-class at the snap | Man/zone at the snap |
|---|---|---|
| Transformer, all coverage defenders | 73.8% | 91.1% |
| Transformer, 5 deepest only (count removed, underneath defenders removed too) | 67.8% | 87.1% |
| GBM, snap features (never sees the count) | 65.6% | 87.3% |
| GBM + the defender count (deliberately leaky) | 66.6% | 87.9% |

- The count is worth about **1 point** to the GBM, so the transformer's +7.2 / +3.2 snap lead over the count-aware GBM is a real architectural gain.
- Within each defender-count group (e.g. the 1,660 test plays with 7 coverage defenders: 71.3% vs 64.5% for 7-class), the transformer still leads.
- The 5-deepest ablation drop comes mostly from losing the alignment of underneath defenders (LBs, nickel), which is real information.
- Fully clean check (all 22 players, no missing rushers): Phase 1E on StatsBomb.

Script: `scratch/phase1c_leak.py` (`--with-count` for the leaky GBM).
