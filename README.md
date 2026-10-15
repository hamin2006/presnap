# presnap

Pre-snap reads from NFL tracking data: the safety shell at the snap, man vs zone, coverage type, post-snap rotation (disguise) and blitz. These feed QB and coach pre-game reports. A video → tracking pipeline comes later, so the same reads can run on ordinary all-22 film.

- Plan: [docs/build_plan.md](docs/build_plan.md) (phases, metrics, go/no-go gates)
- Data: [DATA.md](DATA.md) (sources, joins, caveats) and [docs/data_audit.md](docs/data_audit.md)

## Status

**Phase 0 (foundation): done.** **Phase 1A (shell rule): built; human review deferred** · **Phase 1B (baselines): done** ([scorecard](reports/phase1b_scorecard.md)) · **Phase 1C (coverage model): done** ([results](reports/phase1c_main.md)) · ([reports/phase1a_checks.json](reports/phase1a_checks.json)). Unified coordinate frame, label harmonization, frozen splits, tests, and a v0 opponent report built from charting data ([reports/](reports/)).

## Setup

```bash
uv sync
uv run python download.py            # nflverse / FTN tables (free)
# Kaggle (needs ~/.kaggle token + accepted rules) and StatsBomb downloads: see DATA.md
uv run python -m presnap.build       # -> data/processed/{plays,tracking,player_labels}.parquet (~2 min)
uv run pytest -q
uv run python -m presnap.report_v0   # -> reports/phi_2025_defense_v0.md
```

No raw data is committed. Every source has its own license; see DATA.md.

## Processed tables

| Table | Grain | Notes |
|---|---|---|
| `plays.parquet` | play (`source`, `game_id`, `play_id`) | Situation, formation and personnel, the FTN rush and motion fields, `coverage` / `family` / `man_zone` / `shell_played`, `label_source`, `coverage_ftn`, `labels_agree`, StatsBomb quality flags, `split` |
| `tracking.parquet` | player × 10 Hz frame | `frame` 0 = snap. `x_rel` = yards past the LOS (offense < 0). `y_rel` = yards from mid-field, + = offense's left. `vx`, `vy` in yd/s |
| `player_labels.parquet` | BDB 2026 coverage defender | BDB 2027 `coverage_assignment`, `coverage_assignment_at_snap` and `rushed`. Sparse: use for evaluation, not training |

Splits are frozen in [`splits/splits.csv`](splits/splits.csv). BDB 2026 splits by week: 1–12 train, 13–15 val, 16–18 test. StatsBomb splits by game, 70/15/15.

## Data credits

- NFL Big Data Bowl 2026 and 2027 (NFL Next Gen Stats, via Kaggle). The competition rules govern use.
- StatsBomb American football open data ([statsbomb/amf-open-data](https://github.com/statsbomb/amf-open-data)). **Data provided by StatsBomb.** Use is under the StatsBomb Public Data User Agreement.
- nflverse play-by-play and participation data, and FTN charting data (FTN Data via nflverse, CC-BY-SA 4.0).
