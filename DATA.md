# Data inventory

There are two kinds of data. Play-level charting records *what* the defense played on each play; it's free and covers 2018–2026. Tracking data records *where every player stood and moved*; it comes from the Big Data Bowl on Kaggle and StatsBomb's open data. Everything sits in `data/raw/` (about 2.9 GB, not committed). `uv run python -m presnap.build` turns it into the tables in `data/processed/` described in the README. The audit behind the numbers here is in [docs/data_audit.md](docs/data_audit.md).

## nflverse / FTN (`uv run python download.py`)

| File | Seasons | What's in it |
|---|---|---|
| `participation.parquet` | 2016–2025 | **Coverage type** and **man/zone** for 2018 onward, plus offensive formation, personnel on both sides, defenders in the box, number of pass rushers, routes, time to throw, pressure, and the IDs and positions of all 22 players |
| `ftn_charting.parquet` | 2022–**2026 in-season** | Motion, play action, RPO, screen, number of blitzers, number of pass rushers, box count, backfield count, starting hash, QB alignment, the read the QB threw to |
| `pbp.parquet` | 2016–2025 | nflverse play-by-play: down, distance, field position, score, win probability, EPA, result |
| `nextgen_passing.parquet` | 2016–2025 | Weekly NGS passing aggregates per QB |
| `rosters_weekly.parquet`, `players.parquet` | 2016–2025 | IDs and positions, used for joins |

**Labeled dropbacks by season** (participation joined to pbp, `qb_dropback == 1`):

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| 18,315 | 18,413 | 18,842 | 18,968 | 18,912 | 21,730 | 21,092 | 20,869 |

**These are two different label products. Never pool them.**
- **2018–22 is NGS.** It uses the same 8 classes as BDB: C0/1/2-man, C2/3/4/6 zone, and Prevent.
- **2023+ is FTN.** It adds COMBO, COVER_9 and BLOWN, and drops PREVENT. The class mix also shifts: Cover 3 goes from 31.9% to 18.0%.
- **On shared 2023+ plays, NGS and FTN agree only 57% on the exact coverage**, 84% on shell and 82% on man/zone. FTN calls 28.8% of NGS Cover 3 plays Cover 1.
- In 2023+ participation, `number_of_pass_rushers` comes from FTN and includes junk values such as 44 and 45.
- `presnap/labels.py` maps every raw value to a single vocabulary and records `label_source`.

## Big Data Bowl (Kaggle)

| Competition | Status | What's actually in it |
|---|---|---|
| `nfl-big-data-bowl-2026-analytics` | ✅ `data/raw/bdb2026/` | **Core set.** All of 2023: **14,108 pass plays**.<br>• `train/input_*`: 10 Hz tracking from the snap to the throw. **Frame 1 is the snap** (confirmed on 9,510/9,519 tracks), and the last frame is one before `pass_forward`. Median 26 frames.<br>• `train/output_*`: movement after the throw.<br>• `supplementary_data.csv`: NGS `team_coverage_type`, formation, receiver alignment, box count, play action. It also lists 3,901 plays from 2024 whose tracking was the hidden test set.<br>• **Only coverage defenders, route runners and the passer are tracked**: no OL, no pass rushers. 11 minus the number of tracked defenders equals the charted rusher count on 88% of plays. **Never use which players are tracked as a blitz or rush feature.**<br>• `role = Targeted Receiver` reveals who was thrown to, so don't use it as a pre-throw feature. |
| `nfl-big-data-bowl-2027` | ✅ partly, `data/raw/bdb2027/` | `player_play.csv`: **114,954 plays from the 2023–2025 seasons**, preseason included, with 1–11 rows per play (only the players this competition studies).<br>• Per-player `coverage_assignment` and `coverage_assignment_at_snap`. In 16% of cases the assignment changes after the snap.<br>• `blitzing` means **rushed the passer**, and is true for about 95% of DL. A real blitz is a rusher who isn't DL.<br>• **Sparse on BDB 2026 plays:** only 3,426 of 94,293 tracked coverage defenders (3,163 plays) carry an assignment, and no play is fully labeled. Use these labels for evaluation only.<br>• Tracking: only `game_tracking_2023.csv` is downloaded. It covers 1–6 players per play, starting about 0.9 s before the snap.<br>• `team_coverage_type` agrees with BDB 2026 on 93% of shared plays. |
| `nfl-player-contact-detection` | ✅ rules accepted (Phase 2) | Endzone, Sideline and All29 video synced to NGS tracking. Not downloaded yet. |
| `nfl-health-and-safety-helmet-assignment` | ✅ rules accepted (Phase 2) | Endzone and Sideline video, NGS tracking, helmet boxes. Not downloaded yet. |
| `nfl-big-data-bowl-2025` | ❌ data pulled | Only a README remains. PFF vs NGS label disagreement can't be measured. |
| `nfl-big-data-bowl-2021` | ❌ can't join | The competition is closed. |

Access: put your Kaggle API token in `~/.kaggle/` (`access_token` works with CLI 2.x) and accept each competition's rules on its website. Then run:
`uv run kaggle competitions download -c <slug> -p data/raw/<dir>`

## StatsBomb open data → `data/raw/statsbomb/`

Source: [statsbomb/amf-open-data](https://github.com/statsbomb/amf-open-data), the "TB12DB" release: every Tom Brady game from 2016 to 2022 (124 games). License: StatsBomb Public Data User Agreement. Credit StatsBomb when you publish.

- **Tracking**: 30 Hz, all 22 players on 16,019 of 16,050 plays, derived from broadcast video. Median 3.5 s before the snap.
  - Off-camera positions are **filled in, not left as gaps**: only 0.16% of steps are null. The one quality flag is `on_camera_ratio`, given once per player per play. Safeties are at 0.9 or above on 96% of plays.
  - Track `gsis_player_id` is actually the NFL `nfl_id`, not a GSIS id.
  - `ngs_x = 6x + 90` and `ngs_y = 6y + 30`, so those columns are in sixth-of-a-yard units, not standard NGS yards.
  - **14 plays appear twice** under the same `gsis_play_id`. The build drops them.
- **Plays (`plays/`)**: 172 columns.
  - `play_defensive_back_depths` arrays are **not sorted**.
  - `play_blitz` is never False (mixed TRUE/True casing), so treat null as "no blitz".
  - Shell predicted from the DB-depth field alone is right only 67% of the time.
- **Coverage labels** come from nflverse (NGS, 2018–22). Joining through the tracking JSON gives 6,810 labeled plays after dedup. **5,207 labeled dropbacks** pass the quality filter (1 s or more of pre-snap data, every safety on camera at 0.9 or above with a step at the snap): roughly half have NE/TB on defense, half opponents.
- **Downloading**: the S3 season zips in the README return 404. Use the per-game `url` values in `games.json`:
  `xargs -P 6 -n 1 curl -sfLO < urls.txt`

**What we can and can't cover:** BDB 2026 gives a large sample of the look at the snap and what happens after it. StatsBomb gives pre-snap movement for all 22 players, but always against Brady's offense, and with NE/TB on defense in half the plays. This doesn't fully replace the BDB 2025 set.

## Coordinate frames (handled in `presnap/normalize.py`)

| | x | y | LOS | Direction |
|---|---|---|---|---|
| BDB 2026/2027 | 0–120, end zones included | 0–53.3; increases toward the offense's left when `play_direction` is right | `absolute_yardline_number` | `play_direction` |
| StatsBomb | Goal line to goal line, 0–100 (raw values run from −14 to 112) | **Flipped relative to BDB** | `play_yardline` if `offense_left_to_right`, otherwise `100 − play_yardline` | `offense_left_to_right` |

Both are mapped to `x_rel` (yards past the LOS) and `y_rel` (yards from mid-field, + = offense's left). The y orientation was checked against FTN `starting_hash` for both sources (L ≈ +3.0, R ≈ −3.0), and `tests/test_phase0.py` re-checks it.

## Joins

- BDB `game_id`/`play_id` = nflverse `old_game_id`/`play_id`. Participation stores `old_game_id` as text and `play_id` as a float, so cast both to Int64.
- FTN only carries nflverse string ids. Join through pbp (`game_id` + `play_id`) to get `old_game_id`.
- StatsBomb tracking JSON: `nfl_old_game_id` + `gsis_play_id`. The plays CSV stores its ids as text like "41.0".

## Paid, if we ever need it

| Source | Price | What it adds |
|---|---|---|
| NFL+ Premium (includes NFL Pro) | about $100/yr | **All-22 film** (streaming only), needed for the Phase 1A hand audit |
| PFF+ / PFF Pro | $80 / $200 per yr | Coverage and alignment charting; Pro has an API and CSV exports |
| SIS DataHub | $749/yr | Charting with coverage filters |
| FTN StatsHub / Stats iQ | about $110/yr | Coverage splits, DVOA, pressure |
| Fantasy Points Data Suite | about $200/yr | Route and coverage splits over multiple years |
| SumerSports SumerPass | about $100/yr | Live charting, personnel, pressure |

## Not included (yet)

- **College:** the CollegeFootballData API has play-by-play only. No coverage labels or tracking.
