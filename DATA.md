# Data inventory

There are two kinds of data. Play-level charting tells you *what* the defense played on each play. It's free, it's already downloaded, and it covers 2018–2026. Tracking data tells you *where every player stood and moved* before the snap. It comes from the Big Data Bowl competitions on Kaggle and needs a Kaggle login.

## Downloaded (`uv run python download.py`, about 155 MB in `data/raw/`)

| File | Seasons | What's in it |
|---|---|---|
| `participation.parquet` | 2016–2025 | **Coverage type** (Cover 0/1/2/3/4/6/9, 2-Man, Combo, Prevent, Blown) and **man/zone** for 2018 onward, plus offensive formation, offensive and defensive personnel, defenders in the box, number of pass rushers, routes, time to throw, pressure, and the IDs and positions of all 22 players on the field. Source: NGS through 2022, FTN from 2023. |
| `ftn_charting.parquet` | 2022–**2026 in-season** | Motion, play action, RPO, screen, number of blitzers, number of pass rushers, defenders in the box, offensive backfield count, starting hash, QB alignment, the read the QB threw to |
| `pbp.parquet` | 2016–2025 | nflverse play-by-play with 372 columns: down, distance, field position, score, win probability, EPA, result |
| `nextgen_passing.parquet` | 2016–2025 | Weekly NGS passing aggregates per QB |
| `rosters_weekly.parquet`, `players.parquet` | 2016–2025 | IDs, positions, physical measurements; used to join everything together |

**Labeled dropbacks (coverage type), by season.** These counts come from joining participation to pbp on `qb_dropback == 1`:

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| 18,315 | 18,413 | 18,842 | 19,541 | 18,912 | 21,730 | 21,092 | 20,869 |

That's about **157k labeled dropbacks**. The labels changed source in 2023 (NGS before, FTN after), so check that the two charting providers agree before pooling seasons.

**What this data can't tell you:** where each defender is lined up, safety depth, leverage, or how defenders move before the snap. For those you need the tracking data below.

## Needs a Kaggle login (Big Data Bowl)

Checked against the actual files on 2026-10-15:

| Competition slug | Status | What's really in it |
|---|---|---|
| `nfl-big-data-bowl-2026-analytics` | ✅ downloaded to `data/raw/bdb2026/` (878 MB) | **The core set now.** Tracking in `train/input_2023_w01–18.csv` covers **the full 2023 season, 14,108 pass plays**; `train/output_*.csv` holds movement after the throw. The supplementary file also lists 3,901 plays from 2024, but their tracking was the hidden test set and isn't included. 14,346 of these plays also have per-player labels in BDB 2027. 10 Hz tracking of **every coverage defender, every route runner and the passer**, about 12–13 players per play; no OL, and no pass rushers. Frame 1 appears to be the snap, which gives the alignment at the snap and the movement after it up to the throw; I haven't confirmed frame 1 is exactly the snap. `supplementary_data.csv` adds `team_coverage_type`, man/zone, formation, receiver alignment (e.g. 3x1), defenders in the box, dropback type, play action, routes. **No frames before the snap.** |
| `nfl-big-data-bowl-2027` | ✅ partly downloaded to `data/raw/bdb2027/` | Tracking is **sparse**: only 1–6 players per play (the players the competition is about), starting about 0.9 s before the snap. `player_play.csv` is the valuable file: about 115k plays from 2023–2026 with **per-player `coverage_assignment`, `coverage_assignment_at_snap` and `blitzing`**, plus `team_coverage_type`, lined-up position and motion at the snap. You can join it to BDB 2026 on `game_id` and `play_id` to add per-defender labels. |
| `nfl-big-data-bowl-2025` | ❌ data pulled | Only a README remains. This was the full pre-snap 2022 set. |
| `nfl-big-data-bowl-2021` | ❌ can't join | The competition is closed and the join button is greyed out. |

To set up access:
1. Go to kaggle.com → Settings → **Create New Token**, and save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`.
2. Open each competition page above and click **Join / I understand and accept the rules**. Downloads fail until you've accepted.
3. Then download a competition:
   `uv run kaggle competitions download -c nfl-big-data-bowl-2025 -p data/raw/bdb2025`

## StatsBomb open data (free; download links in the GitHub repo README) → `data/raw/statsbomb/`

Source: [statsbomb/amf-open-data](https://github.com/statsbomb/amf-open-data). The "TB12DB" release covers **every Tom Brady game from 2016 to 2022**: 124 games, Patriots then Buccaneers.

- **Tracking (`tracking/`)**: 30 Hz, **all 22 players**, positions derived from broadcast video. Each step has a `time_since_snap` value, and plays start about **3.4 s before the snap**, so this is the only full-field pre-snap sequence we have. It also includes NGS-scale coordinates and per-player `on_camera_ratio`; players off camera have gaps.
- **Plays (`plays/`)**: 172 columns per play. Includes **defensive back depths, linebacker depths**, initial formation versus formation at the snap (shifts), motion type, blitz, simulated pressure, drop eight, box count, personnel, pocket geometry and catch probability.
- **Events (`events/`)**: per-event data with freeze frames.
- **No coverage-type column.** Join to nflverse `participation` on `gsis_old_game_id` + `gsis_play_id`; cast both keys to Int64, because about 6k plays have `NA` ids. Checked: 18,116 plays, 11,851 joined, and **4,013 dropbacks with a coverage label (2018–2022)**. Both teams' offensive plays are included, so we get Brady's offense against many defenses and the Patriots/Buccaneers defense against many offenses.
- **Downloading tracking:** the season zips on S3 linked in the README fail (checked 2026-10-15). Use the per-game `url` values in `games.json` instead: `xargs -P 6 -n 1 curl -sfLO < urls.txt`.
- **Caveats:** the offense is always Brady's, but the defenses vary. Licensing is the StatsBomb Public Data User Agreement (`doc/LICENSE.pdf` in the repo): credit StatsBomb when publishing.

**The gap, mostly closed:** BDB 2026 gives a large sample of the **look at the snap**, the movement afterward and the coverage played. StatsBomb gives a smaller sample with **pre-snap movement for all 22 players** (motion, safety rotation before the snap). Together they cover what the BDB 2025 set would have.

## Paid, if we ever need it

| Source | Price | What it adds |
|---|---|---|
| NFL+ Premium (includes NFL Pro) | about $100/yr | **All-22 coaches film** (streaming only) and the NGS-powered NFL Pro platform. Use it to check labels by eye; there are no bulk exports. |
| PFF+ / PFF Pro | $80 / $200 per yr | Coverage and alignment charting, grades. Pro has an API and CSV exports. |
| SIS DataHub | $749/yr | Charting with coverage filters, including by route and coverage |
| FTN StatsHub / Stats iQ | about $110/yr | Coverage splits, DVOA, pass-rush and pressure charting |
| Fantasy Points Data Suite | about $200/yr | Route and coverage splits over multiple years |
| SumerSports SumerPass | about $100/yr | Live charting, personnel, pressure |

Free summary tables (Sharp Football coverage rates, StatRankings) are useful for checking our team-level numbers.

## How the pieces join

- Big Data Bowl tracking uses NFL `gameId` and `playId`. nflverse uses `old_game_id` (the same 10-digit ID) and `play_id`.
- 2022 weeks 1–9 has **both** PFF labels (Big Data Bowl 2025) and NGS labels (nflverse). That's a free way to measure how often charters disagree, which sets the ceiling on how accurate any coverage model can look.

## Not included (yet)

- **College:** the CollegeFootballData API (free key) has play-by-play only. No coverage labels or tracking.
- **PFF and FTN premium:** paid. This mostly duplicates the coverage and blitz labels we already have.
