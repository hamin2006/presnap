# Phase 1B baseline scorecard

BDB 2026 (2023 season, NGS labels): train weeks 1–12, early-stop on 13–15, **test on 16–18**. Accuracy with a 95% game-clustered bootstrap CI; `agree` = accuracy on test plays where NGS and FTN give the same label.

## man

| Model | n | Accuracy | 95% CI | Log loss | Macro F1 | Acc (agree) |
|---|---|---|---|---|---|---|
| majority | 2455 | **71.8%** | 69.5%–74.0% | 0.595 | 0.418 | 59.6% |
| situation | 2455 | **75.4%** | 73.4%–77.7% | 0.519 | 0.623 | 67.3% |
| gbm_snap | 2455 | **87.7%** | 86.2%–88.8% | 0.297 | 0.843 | 85.3% |
| gbm_post_2s | 2455 | **91.5%** | 90.1%–92.9% | 0.207 | 0.893 | 91.0% |
| gbm_throw | 2455 | **90.2%** | 89.1%–91.5% | 0.217 | 0.875 | 89.3% |

## cov7

| Model | n | Accuracy | 95% CI | Log loss | Macro F1 | Acc (agree) |
|---|---|---|---|---|---|---|
| majority | 2451 | **32.6%** | 30.1%–35.1% | 1.702 | 0.070 | 26.6% |
| situation | 2451 | **38.8%** | 36.7%–41.0% | 1.552 | 0.234 | 38.6% |
| gbm_snap | 2451 | **65.7%** | 63.3%–68.0% | 0.938 | 0.619 | 69.1% |
| gbm_post_2s | 2451 | **81.1%** | 79.3%–82.7% | 0.484 | 0.782 | 84.9% |
| gbm_throw | 2451 | **79.0%** | 77.2%–80.5% | 0.561 | 0.756 | 83.3% |

## shell

| Model | n | Accuracy | 95% CI | Log loss | Macro F1 | Acc (agree) |
|---|---|---|---|---|---|---|
| majority | 2451 | **54.4%** | 51.6%–57.0% | 0.842 | 0.235 | 59.0% |
| situation | 2451 | **58.0%** | 56.0%–60.1% | 0.769 | 0.488 | 61.0% |
| gbm_snap | 2451 | **83.5%** | 81.5%–85.4% | 0.416 | 0.801 | 85.8% |
| gbm_post_2s | 2451 | **92.8%** | 91.6%–94.0% | 0.183 | 0.898 | 95.4% |
| gbm_throw | 2451 | **91.0%** | 89.5%–92.5% | 0.233 | 0.882 | 93.6% |

## StatsBomb clean check (all 22 tracked, 2018–22, split by game)

| Target | Model | n | Accuracy | 95% CI | Log loss |
|---|---|---|---|---|---|
| man | majority | 809 | **63.0%** | 54.1%–71.6% | 0.659 |
| man | situation | 809 | **73.2%** | 67.8%–78.8% | 0.555 |
| man | gbm_snap | 809 | **87.6%** | 85.6%–89.6% | 0.293 |
| cov7 | majority | 805 | **26.8%** | 18.2%–37.0% | 1.773 |
| cov7 | situation | 805 | **37.0%** | 33.0%–41.6% | 1.565 |
| cov7 | gbm_snap | 805 | **65.1%** | 60.8%–69.5% | 0.952 |
| shell | majority | 805 | **50.7%** | 44.1%–57.2% | 0.929 |
| shell | situation | 805 | **59.2%** | 55.2%–63.7% | 0.778 |
| shell | gbm_snap | 805 | **80.5%** | 77.4%–83.3% | 0.443 |
