# Local data audit: what the downloaded football datasets can and cannot support (tracking to coverage/shell/blitz model)

All numbers below come from running code against the files in `presnap/data/raw/` on 2026-10-15. Nothing in `data/raw/` was modified. The scripts and their full printed output are kept in `presnap/scratch/audit/`, and each finding cites the script that produced it. `common.py` holds the loaders. Parquet caches of the big CSVs are written to `scratch/audit/cache/` (157 MB, safe to delete). To rerun everything: `cd presnap && for f in scratch/audit/a0*.py; do uv run python $f; done`. a05 has to run before a06.

Script index (source links used below):
- [a01_bdb2026.py](../scratch/audit/a01_bdb2026.py) / [a01_out.txt](../scratch/audit/a01_out.txt): BDB 2026 structure, frames, roles, labels, rusher-count implication
- [a02_snap_alignment.py](../scratch/audit/a02_snap_alignment.py) / [a02_out.txt](../scratch/audit/a02_out.txt): BDB 2026 frames matched to BDB 2027 `ball_snap` / `pass_forward` events
- [a03_bdb2027_labels.py](../scratch/audit/a03_bdb2027_labels.py) / [a03_out.txt](../scratch/audit/a03_out.txt): BDB 2027 per-player labels, joins, and cross-source confusion matrices
- [a04_nflverse.py](../scratch/audit/a04_nflverse.py) / [a04_out.txt](../scratch/audit/a04_out.txt): nflverse/FTN distributions, taxonomy, team-season sizes, and grouped agreement
- [a05_statsbomb_tracks.py](../scratch/audit/a05_statsbomb_tracks.py): per-track StatsBomb summary extraction
- [a06_statsbomb_audit.py](../scratch/audit/a06_statsbomb_audit.py) / [a06_out.txt](../scratch/audit/a06_out.txt): StatsBomb visibility, pre-snap duration, label join, depth and blitz checks
- [a07_misc.py](../scratch/audit/a07_misc.py) / [a07_out.txt](../scratch/audit/a07_out.txt): BDB 2026 per-week counts and drift, StatsBomb coordinate ranges and null steps
- [a08_targets.py](../scratch/audit/a08_targets.py) / [a08_out.txt](../scratch/audit/a08_out.txt): label-target class counts on the tracking plays

## BDB 2026: is frame 1 the snap, how long are plays, which defenders are present, and what is the label distribution?

### Takeaway
Frame 1 is the snap, confirmed against BDB 2027's `ball_snap` event on 9,510 of 9,519 matched player-plays. The last input frame is the frame just before `pass_forward`, so the input runs from the snap to the throw: median 26 frames (2.6 s) at 10 Hz. Only non-rushing defenders are tracked: every defensive row has role "Defensive Coverage", usually 7 per play. 11 − (number of coverage defenders) equals the charted pass-rusher count on 88% of plays. So the roster of who is tracked leaks the rush and blitz answer, and pass rushers are never visible.

### Cited Findings
- Size: 4,880,579 input rows, 14,108 plays, 272 games (all of 2023 REG weeks 1–18; every team plays 17 games). `supplementary_data.csv` has 18,009 plays (14,108 from 2023 plus 3,901 from 2024 that have no tracking), no duplicate keys, and every tracked play has a supplementary row. — [a01_out.txt](../scratch/audit/a01_out.txt), [a07_out.txt](../scratch/audit/a07_out.txt)
- **Frame 1 equals the snap (confirmed):** for 9,519 (game, play, nfl_id) tracks present in both BDB 2026 and BDB 2027 tracking (7,314 plays), every BDB 2026 frame 1–5 position has an exact x,y match in BDB 2027. The matched BDB 2027 frame is the `ball_snap` frame for 9,510. The offset is +1 frame for 4, and the snap is undetermined for 5. Restricted to moving players with a unique position match, the result is 5,857 at offset 0 versus 4 at +1. — [a02_out.txt](../scratch/audit/a02_out.txt)
- **The last input frame is one frame before the throw:** the last BDB 2026 input frame minus the BDB 2027 `pass_forward` frame is −1 for all 9,514 matched tracks. Input frame count equals (pass_forward frame − snap frame) exactly for 9,510 tracks and is off by one for 4. — [a02_out.txt](../scratch/audit/a02_out.txt)
- Kinematics agree. At frame 1, route runners have median speed 0.01 yd/s and 89.2% have s < 0.3. The still share falls to 79.2% at frame 2, 33.4% at frame 4 and 0.46% at frame 10. The passer is still in 92.3% of plays at frame 1, and the median passer |Δx| from frame 1 to 2 is 0.01 yd. Coverage defenders are already moving at frame 1 (median 0.28 yd/s, 51.2% still), which fits pre-snap movement and creep. — [a01_out.txt](../scratch/audit/a01_out.txt)
- Frames per play (snap to throw): min 8, p5 17, p25 22, median 26, p75 32, p95 45, max 123, mean 28.1 (2.81 s). 465 plays have ≥ 50 frames and 21 have ≥ 80. Post-throw `num_frames_output`: median 10, p5 6, p95 23, max 94. Every player has every frame (no gaps), and the input file has zero nulls in every column. — [a01_out.txt](../scratch/audit/a01_out.txt)
- Roles at frame 1: Defensive Coverage 94,293; Other Route Runner 50,644; Targeted Receiver 14,108 (exactly 1 per play); Passer 14,105 (3 plays have none). "Defense side but not coverage role" has 0 rows, so every tracked defender is a coverage defender. OL and pass rushers are absent. — [a01_out.txt](../scratch/audit/a01_out.txt)
- Coverage defenders per play: 7 → 9,547 plays (67.7%), 6 → 2,950, 5 → 826, 8 → 597, 4 → 151, 3 → 14, 9 → 14, 11 → 6, 10 → 2, 0 → 1. Total tracked players per play ranges from 1 to 17, mode 13 (7,369 plays). Route runners per play: 5 → 9,421, 4 → 3,708, 3 → 859. Participation `n_defense` is 11 on all 14,108 plays, so the missing defenders are the rushers. — [a01_out.txt](../scratch/audit/a01_out.txt)
- Positions of coverage defenders: CB 37,500; FS 17,018; SS 13,764; ILB 10,403; OLB 7,297; MLB 7,064; DE 596; S 481; DT 115; NT 45 (dropping linemen); FB 9. — [a01_out.txt](../scratch/audit/a01_out.txt)
- **11 − n_coverage vs charted rushers:** exact match with participation `number_of_pass_rushers` on 88.0% of plays (98.3% within ±1) and with FTN `n_pass_rushers` on 88.1% (98.4% within ±1), n = 14,104 joined. Diagonal: 4 rushers implied, 9,141 agree; 5, 2,206; 6, 520; 3, 459. The largest off-diagonals are implied 5 vs charted 4 (595) and implied 4 vs charted 5 (257). — [a01_out.txt](../scratch/audit/a01_out.txt)
- BDB 2027 shows the excluded players directly. Of the BDB 2027 defender rows on BDB 2026 plays that are absent from BDB 2026 tracking (2,743), 1,693 are EDGE and 975 INTERIOR_LINE; 2,742 have `blitzing=True` and 2,695 have `coverage_assignment=NO_ASSIGNMENT`. — [a03_out.txt](../scratch/audit/a03_out.txt)
- `player_to_predict` is True for all 14,108 targeted receivers and 31,937 of 94,293 coverage defenders. The output files hold post-throw x,y only for those players: 1–9 per play, mostly 2–4. — [a01_out.txt](../scratch/audit/a01_out.txt)
- `team_coverage_type` on the 14,108 tracked plays: COVER_3_ZONE 4,482 (31.8%), COVER_1_MAN 3,175 (22.5%), COVER_4_ZONE 2,346 (16.6%), COVER_2_ZONE 1,837 (13.0%), COVER_6_ZONE 1,388 (9.8%), COVER_0_MAN 601 (4.3%), COVER_2_MAN 237 (1.7%), PREVENT 39 (0.3%), null 3. Man/zone: ZONE 10,092, MAN 4,013, null 3. The 2024 label-only plays: C3 1,182, C1 933, C2Z 681, C4 514, C6 305, C0 180, C2M 95, PREVENT 9, null 2. — [a01_out.txt](../scratch/audit/a01_out.txt)
- Split by down and distance (row %): on 1st & 7–10 (n = 4,778), C3 is 38.3%, C4 18.8%, C1 16.3%, C2Z 13.1%, C6 10.4%, C0 1.9%. On 3rd & 1–3 (n = 766), C1 is 41.1%, C3 25.5%, C0 10.3%. On 3rd & 4–6 (n = 1,133), C1 40.2%, C3 23.2%, C0 5.7%. On 3rd & 11+ (n = 678), C3 33.5%, C2Z 19.9%, C4 16.4%, C6 11.9%, C1 10.8%. On 4th & 1–3 (n = 184), C1 43.5% and C0 19.6%. On 1st & 1–3 (goal-to-go, n = 57), C0 is 77.2%. On 2nd & 11+ (n = 909), C4 25.3%. — [a01_out.txt](../scratch/audit/a01_out.txt)
- Other supplementary fields on the tracked plays: pass_result C 9,738, I 4,032, IN 338 (all are targeted passes; no sacks or scrambles). `play_nullified_by_penalty` = Y on 1 play. `defenders_in_the_box` has mode 6 (7,031 plays), with 5 on 2,703 and 7 on 2,763. — [a01_out.txt](../scratch/audit/a01_out.txt)

### Inferences
- BDB 2026 can support a **coverage-at-snap** model: coverage defenders' alignment and velocity at frame 1, plus receivers and passer. It can also support a much stronger **coverage-recognition-after-snap** model on frames 1–N, which is the classic BDB-style post-snap classifier. It cannot show any defender's motion before the snap.
- Do not use the number or identity of tracked defenders as a feature for any blitz or rusher target. Rusher count is almost a deterministic function of n_coverage (88% exact), and the role field is assigned after the play. For blitz and pressure targets, BDB 2026 at frame 1 cannot see the rushers at all. Even for coverage targets, n_coverage correlates with Cover 0 (more rushers), so the model should be checked with and without it.
- Every play ends at a throw, so the sample is biased toward completed dropbacks (no sacks, scrambles or throwaways without a target). Any frame-N feature also has a variable cutoff, from 0.8 s to 12.3 s.
- Coverage varies strongly with game situation: Cover 1 and Cover 0 on 3rd/4th & short, two-high on long yardage. A model given only situation features will look strong, so report a situation-only baseline beside any tracking model.

### Gaps
- Exact sub-frame timing of the snap within the 100 ms frame was not checked. The frame matches BDB 2027's snap event, which is itself an NGS event tag.
- I did not check whether to filter outlier plays such as spikes, kneels and trick plays. There are 6 plays with 11 coverage defenders and 46 with 0 charted rushers.

## BDB 2027: what the per-player labels contain, how they join to BDB 2026, and whether coverage labels agree across charting sources

### Takeaway
`player_play.csv` is **sparse**: it has rows only for the 1–11 players tracked in the BDB 2027 competition (mean about 2.7 per play), not all 22. Only 3,426 of 94,293 BDB 2026 coverage defenders (3,163 plays) get a `coverage_assignment`, and no play has all its defenders labeled. `blitzing` means "rushed the passer", not "blitzed". Team-level labels agree with BDB 2026 on 93.1% of plays. Against FTN they agree only 56.6–57.3% exactly, 83–84% on shell, and 81–82% on man/zone.

### Cited Findings
- 314,197 rows; seasons 2023 (31,720 plays, 333 games), 2024 (40,505), 2025 (42,729), for 114,954 plays in total. This includes preseason: 5,512 / 6,202 / 6,345 plays. Game dates run into Jan–Feb 2026, but the last season is 2025. — [a03_out.txt](../scratch/audit/a03_out.txt), [games.csv check](../scratch/audit/a03_bdb2027_labels.py)
- Rows per play: 1 → 29,845 plays, 2 → 29,057, 3 → 23,826, 4 → 16,202, 5 → 9,454, 6+ → 6,570. Defensive rows per play: 0 → 26,152, 1 → 45,621, 2 → 26,906, 3 → 12,218, 4+ → 4,057. On 2023 the rows map one-to-one to the players in `game_tracking_2023.csv`: 50,911 tracked (play, player) pairs versus 50,755 player_play rows. — [a03_out.txt](../scratch/audit/a03_out.txt), [a08_out.txt](../scratch/audit/a08_out.txt)
- `coverage_assignment` values over all seasons: null 241,545; NO_ASSIGNMENT 29,442; MAN 10,745; CURL_FLAT_LEFT 2,988; DEEP_THIRD_RIGHT 2,859; CURL_FLAT_RIGHT 2,781; DEEP_THIRD_MIDDLE 2,655; DEEP_THIRD_LEFT 2,065; QUARTERS_OUTSIDE_RIGHT 2,042; QUARTERS_INSIDE_LEFT 2,036; QUARTERS_INSIDE_RIGHT 1,771; FLAT_RIGHT 1,718; DEEP_HALF_LEFT 1,692; DEEP_HALF_RIGHT 1,481; DEEP_FREE 1,469; HOOK_CURL_LEFT 1,403; FLAT_LEFT 1,361; HOOK_CURL_RIGHT 1,358; QUARTERS_OUTSIDE_LEFT 1,335; HOLE 1,274; PREVENT 177. `coverage_assignment_at_snap` uses the same 20 values with similar counts. — [a03_out.txt](../scratch/audit/a03_out.txt)
- Labeled defender rows by season: 2023, 9,471 of 21,420; 2024, 25,279 of 53,985; 2025, 37,902 of 77,747. Offense rows never carry a coverage label. — [a03_out.txt](../scratch/audit/a03_out.txt)
- The assignment changes between the snap and the final label for 16.0% of defenders. The top changes are DEEP_THIRD_RIGHT→MAN (404), QUARTERS_OUTSIDE_RIGHT→DEEP_THIRD_RIGHT (362), QUARTERS_OUTSIDE_RIGHT→FLAT_RIGHT (270) and MAN→DEEP_THIRD_LEFT (264). These are rotations and disguises after the snap. — [a03_out.txt](../scratch/audit/a03_out.txt)
- `blitzing`: False 163,754; True 36,973; null 113,470. Among defenders it is True for 92.3% of EDGE, 98.3% of INTERIOR_LINE, 11.8% of MLB, 9.6% of OLB, 6.1% of SLOT_CB, 0.4% of CB and 0.1% of HIGH_SAFETY. Offense rows are never True. — [a03_out.txt](../scratch/audit/a03_out.txt)
- `lined_up_position` (defense): EDGE, CB, INTERIOR_LINE, HIGH_SAFETY (25,608), SLOT_CB, OLB, MLB, BOX_SAFETY (266), plus 839 nulls. It is a role at the snap, more useful than roster position for safeties. — [a03_out.txt](../scratch/audit/a03_out.txt)
- **Join to BDB 2026:** 10,620 of 14,108 tracked plays have at least one player_play row. 14,346 of all 18,009 supplementary plays, including 2024, have a row; this is where DATA.md's "14,346" comes from. Only 3,426 of 94,293 BDB 2026 coverage defenders have a `coverage_assignment`. By play: 0 labeled defenders on 10,944 plays, 1 on 2,900, 2 on 263, and all defenders on 0. Their labels are MAN 849, DEEP_THIRD_RIGHT 330, CURL_FLAT_LEFT 258, and so on. — [a03_out.txt](../scratch/audit/a03_out.txt)
- `game_tracking_2023.csv`: 3,665,443 rows, 31,690 plays, 333 games. Players per play: 1 → 17,738, 2 → 9,662, 3 → 3,417, 4 → 774, 5 → 92, 6 → 7. It has an ISO `time` and no frame_id, with frames 100 ms apart (3 gaps of 200 ms). Frames before `ball_snap`: median 9, p5 9, max 10, so about 0.9 s before the snap. 31,603 plays have a `ball_snap` event. Event vocabulary includes line_set (1,610), man_in_motion (1,827), shift (397), play_action and pass_forward. Of the tracked (play, player) pairs, 21,269 are defenders; 9,471 have a coverage_assignment and 5,632 have blitzing=True. — [a02_out.txt](../scratch/audit/a02_out.txt), [a08_out.txt](../scratch/audit/a08_out.txt)
- **BDB 2026 vs BDB 2027 `team_coverage_type`, same plays:** n = 14,339, agreement 93.07%. The top disagreements (BDB26 → BDB27) are COVER_1_MAN→COVER_3_ZONE 298, COVER_4→COVER_3 126, COVER_6→COVER_4 73, COVER_2Z→COVER_6 58 and COVER_1→COVER_0 58. Within BDB 2027 no play has conflicting team labels across rows. — [a03_out.txt](../scratch/audit/a03_out.txt)
- **NGS-taxonomy labels (BDB) vs nflverse participation `defense_coverage_type` (FTN, 2023+):** joined on old_game_id + play_id. Exact taxonomy match (COVER_N_X↔COVER_N, COVER_2_MAN↔2_MAN): BDB26 vs FTN, n = 17,971, 56.63% (2023 55.81%, 2024 59.58%); BDB27 vs FTN, n = 56,889, 57.30% (2023 53.03%, 2024 54.26%, 2025 63.53%). — [a03_out.txt](../scratch/audit/a03_out.txt)
- The confusion is structured, not random. 1,627 of the 5,655 BDB26 COVER_3_ZONE plays (28.8%) are FTN COVER_1, and 308 are FTN COMBO. Of BDB26 COVER_6, 516 of 1,693 are FTN COVER_2. Of BDB26 COVER_4, 389 are FTN COVER_2, 327 FTN COVER_1 and 309 FTN 2_MAN. Of BDB26 COVER_0, only 460 of 781 are FTN COVER_0 (186 are FTN COVER_1). — [a03_out.txt](../scratch/audit/a03_out.txt)
- Grouped agreement: shell (zero / single-high = 1, 3 / two-high = 2, 2-man, 4, 6; excluding FTN COMBO, 9, BLOWN and PREVENT, which are 6.2% of joined plays): BDB26 83.43% (n = 16,844), BDB27 84.11% (n = 54,236; 2023 82.3%, 2024 83.8%, 2025 85.7%). Man/zone: BDB26 80.95% (n = 17,971), BDB27 81.98% (n = 56,889). Disagreements are asymmetric: NGS zone with FTN man 2,915, versus NGS man with FTN zone 509 (BDB26). — [a04_out.txt](../scratch/audit/a04_out.txt)

### Inferences
- BDB 2027 per-player labels **cannot** serve as a dense per-defender training target on BDB 2026 tracking: 3.6% of defenders and never a full play. They work as (a) a sparse auxiliary or eval set of about 3.4k defender assignments with full-play context from BDB 2026, and (b) a separate set of about 9.5k labeled defenders in 2023, with 0.9 s of pre-snap tracking but almost no teammates. Labeled defenders rise to about 25k and 38k in 2024–25, but only 2023 tracking was downloaded.
- To label blitzes from BDB 2027, use `blitzing=True AND lined_up_position NOT IN (EDGE, INTERIOR_LINE)`. Do not use the raw `blitzing` field.
- `coverage_assignment_at_snap` vs the final label (16% changed) directly measures disguise and rotation after the snap. It is a natural target for a "what will this defender do" model.
- Label-noise ceiling: two NGS-family label versions agree 93%, and NGS and FTN agree 57% on the 8–10 class label, 84% on shell and 82% on man/zone. A model evaluated against FTN labels while trained on NGS labels, or the reverse, will look much worse than it is. Pick one source per target.

### Gaps
- Not determined: why BDB 2026 and BDB 2027 labels differ on 7% of plays (a later re-chart, or a different NGS model version). No documentation is available locally.
- Only `game_tracking_2023.csv` is local. Whether 2024 and 2025 sparse tracking exists in the Kaggle competition was not checked, since no new data could be downloaded.

## StatsBomb: players tracked, safety visibility before the snap, pre-snap time, role mapping, and overlap with coverage labels

### Takeaway
StatsBomb gives all 22 players on 16,019 of 16,050 plays, with a median of 3.5 s before the snap. Off-camera positions are not left as gaps. `on_camera_ratio` is a per-track quality flag, and safeties are on camera ≥ 0.9 of the track 96% of the time. Roles map to roster positions through `gsis_player_id` = NFL `nfl_id` (2,467 of 2,471 ids). 5,215 labeled dropbacks have ≥ 1 s of pre-snap data and all safeties at ≥ 0.9 on-camera. Joining through the tracking JSON gives 6,822 labeled dropbacks, versus 4,013 through the plays CSV.

### Cited Findings
- 124 game files (the 125th entry in `tracking/` is `urls.txt`) hold 16,050 plays, by season: 2016/17 2,308; 2017/18 2,095; 2018/19 2,060; 2019/20 2,157; 2020/21 2,527; 2021/22 2,529; 2022/23 2,374. Play types: Pass 10,058, Rush 5,941, 2-pt 51. — [a06_out.txt](../scratch/audit/a06_out.txt)
- Tracks per play: 22 on 16,019 plays, 21 on 26, and fewer on 5. Defensive tracks per play: 11 on 15,992, 10 on 26, and 22 on 15 (a team-id mismatch on those plays). The undocumented field `player_coverage_count` is 22 on only 10,715 plays (66.8%). — [a06_out.txt](../scratch/audit/a06_out.txt)
- `position_code` is coarse (DB 79,476 / LB 48,096 / DL 49,022 / OL / WR / TE / RB / QB) and does not separate CB from S. Each track's `player` dict has player_id, name, position_code, jersey_number and gsis_player_id. `gsis_player_id` is the NFL nfl_id (Brady = 25511), not the 00-00xxxxx GSIS id. Joined to `players.parquet` nfl_id, DB tracks map to CB 35,286, FS 13,568, S 12,794, SAF 7,960, generic DB 8,855 and other 1,013 (LB 678, OT 79, OLB 78, …). 42 tracks have a null gsis_player_id. — [a06_out.txt](../scratch/audit/a06_out.txt) and in-session check
- Roster-identified safeties per play: 1 → 2,462, 2 → 6,769, 3 → 3,577, 4 → 750. Roster position is not the alignment role, so "safety" here means a player listed as FS/S/SAF. — [a06_out.txt](../scratch/audit/a06_out.txt)
- **Pre-snap time** (−min `time_since_snap`), overall: p0 23.6 s, p5 9.3 s, median 3.5 s, p75 1.86 s, p95 0.47 s. 13 plays start after the snap. By season the median is 3.8/3.9/3.9/4.2/3.8 s for 2016–2020 but 2.8 s (2021) and 3.2 s (2022). The share with < 1 s pre-snap is 10–13% for 2016–2020 and 19.2% / 15.7% for 2021 and 2022. — [a06_out.txt](../scratch/audit/a06_out.txt)
- **Visibility:** safety `on_camera_ratio` p1 0.772, p5 0.928, p10 0.993, median 1.0. The share of tracks below 0.9 is 3.8% for safeties, 6.1% for CBs, 5.7% for LBs, 8.0% for DL and 9.6% for offense. It is worst in the 2020/21 season (DL 13.2%, offense 14.0%). Plays where every roster safety has ≥ 0.9: 12,649 of 13,587. Plays where every DB has ≥ 0.9: 11,722 of 13,884. — [a06_out.txt](../scratch/audit/a06_out.txt)
- **Off-camera steps are not gaps.** Pre-snap steps with an x,y position are present for essentially every track (median 1.0 of expected 30 Hz steps; only 0.10% of tracks have < 95%). Null x,y occurs on only 0.16% of all steps. So positions while a player is off camera appear to be imputed, and nothing per step marks which steps are imputed. The only per-step flag, `calibration_fault`, is present on 801 of 31,039 sampled tracks. 346 plays have `calibration_fault_ratio` > 0 (max 0.115). — [a06_out.txt](../scratch/audit/a06_out.txt), [a07_out.txt](../scratch/audit/a07_out.txt)
- Coordinates: 30 Hz, and SB x ranges from −14.0 to 112.2 and y from −7.6 to 59.0 (yards, a static camera-side frame, not normalized). `offense_left_to_right` gives the direction. `ngs_x = 6·x + 90` and `ngs_y = 6·y + 30` exactly (std ≈ 1e-14 on 97k steps), so the "NGS" columns are 1/6-yard units with an offset, not standard NGS yards. — [a07_out.txt](../scratch/audit/a07_out.txt), in-session regression
- **Coverage labels:** joining the tracking JSON's `nfl_old_game_id` + `gsis_play_id` straight to participation gives 6,822 labeled plays, all dropbacks, by NFL season: 2018 1,177; 2019 1,191; 2020 1,548; 2021 1,502; 2022 1,404. 5,882 of these have ≥ 1 s pre-snap. 5,215 also have every roster safety at ≥ 0.9 on-camera with a step at the snap: 2,620 with NE/TB on defense and 2,595 with opponents on defense. Of those, 4,749 have every DB at ≥ 0.9. Classes of the 5,215: C1 1,516, C3 1,416, C2 745, C4 716, C6 319, C0 301, 2-MAN 183, PREVENT 19. — [a06_out.txt](../scratch/audit/a06_out.txt)
- The plays CSVs cover only 2017–2022 (18,116 rows, no 2016). Tracking joins the plays CSV for 13,640 of 16,050 plays. `gsis_play_id` is stored as text like "41.0", with 31 null play ids and 1 null game id. — [a06_out.txt](../scratch/audit/a06_out.txt)
- **`play_defensive_back_depths` vs nflverse shell:** stored as Postgres array strings "{3.92,15.74,…}", **not sorted** despite the dictionary saying ascending, with 10 empty strings. DBs per array: 5 on 4,045 plays, 6 on 1,287, 4 on 1,090. A rule counting DBs ≥ 8 yd deep gives 67.0% shell accuracy (≥ 10 yd: 64.5%; ≥ 12 yd: 51.1%), against a 55.1% majority baseline (n = 6,704). With ≥ 10 yd, single-high plays split 0 deep: 458, 1: 2,515, 2: 653; two-high plays split 0: 185, 1: 985, 2: 1,349. — [a06_out.txt](../scratch/audit/a06_out.txt)
- **`play_blitz`** takes only TRUE/True or NA/null, never an explicit False. TRUE is upper-case in 2017–2020 files and "True" in 2021–22 files. The field is entirely null in 2016 and 63–71% null in 2021–22. Treating non-True as no blitz: against participation rushers ≥ 5, agreement is 86.0% for 2017–2020 (n = 5,520; SB blitz rate 24.9%) and 90.6% for 2021–22 (n = 3,128). Against FTN `n_blitzers > 0` on 2022 plays, agreement is 82.0% (n = 1,512). On those same plays, NGS rushers ≥ 5 vs FTN blitz agree 85.8%. — [a06_out.txt](../scratch/audit/a06_out.txt)

### Inferences
- StatsBomb is the only local source for a **pre-snap-movement** model covering all 22 players: safety rotation, creep, motion response. Expect about 5.2k labeled dropbacks after quality filters, half of them with the same defense (NE/TB) and all with the same QB's offense. That is enough for fine-tuning or validating a model pretrained on BDB 2026 snap alignments, but thin for training a 7–8-class model from scratch.
- Because off-camera positions are imputed silently, `on_camera_ratio` should be used as a filter (≥ 0.9 on DBs). Imputed segments may look smooth but be wrong, and safeties at the top of the broadcast frame are the most likely to be imputed.
- Depth-only rules reach about 65–67% on shell, so the depth arrays add little. Compute safety depth and width from tracking at the snap instead.
- StatsBomb `play_blitz` is usable as a blitz target for 2017–2020 (~86% agreement with rushers ≥ 5, about the same as NGS-vs-FTN), but missing values must be read as "no blitz".

### Gaps
- No per-step on-camera flag exists, so pre-snap visibility of a safety cannot be separated from post-snap visibility. Whether imputed segments can be detected from motion was not resolved: in a 1/12-file sample, the share of consecutive identical positions did not track on_camera_ratio.
- The meaning of `player_coverage_count` is undocumented in the local dictionary.
- The SB x origin relative to the goal line was not pinned down. x runs −14 to 112, so it is likely not a 0–120 frame; check against `play_yardline` before mixing with BDB coordinates.
- The 2016 tracking (2,308 plays) has no plays-CSV row and no nflverse coverage label (participation labels start in 2018).

## nflverse/FTN: coverage distribution by season and source era, taxonomy change, and team-season sample sizes

### Takeaway
Coverage labels exist for 2018–2025 (none for 2016–17). In 2023 both the source (NGS → FTN) and the taxonomy change: COMBO, COVER_9 and BLOWN are added and PREVENT is dropped. The class mix shifts sharply, e.g. Cover 3 goes from 31.9% to 18.0% and 2-man from 1.8% to 6.1%. The NGS-sourced labels (participation 2018–22, BDB 2026, BDB 2027) form one consistent taxonomy, and FTN 2023+ is a different, noisier one. Team-season samples are 478–734 labeled dropbacks, enough for main-class tendencies but not for rare classes by situation.

### Cited Findings
- Labeled dropbacks (REG + POST; qb_dropback = 1): 2018 18,315; 2019 18,413; 2020 18,842; 2021 18,968; 2022 18,912; 2023 21,730; 2024 21,092; 2025 20,869. Label coverage is 86.4–89.3% of dropbacks in 2018–22 and 99.6–99.9% in 2023–25. DATA.md's 2021 figure (19,541) does not match the 18,968 found here. — [a04_out.txt](../scratch/audit/a04_out.txt)
- Coverage %, 2018 → 2022 (NGS) and 2023 / 2024 / 2025 (FTN): COVER_3 34.2 → 31.9 | 18.0 / 17.6 / 27.3; COVER_1 28.9 → 22.7 | 32.1 / 36.5 / 22.5; COVER_2 12.4 → 14.2 | 15.7 / 20.6 / 23.0; COVER_4 11.1 → 16.1 | 12.1 / 6.2 / 10.1; COVER_6 6.5 → 8.6 | 5.1 / 3.5 / 5.5; COVER_0 3.8 → 4.2 | 3.3 / 4.5 / 3.9; 2_MAN 2.5 → 1.8 | 6.1 / 7.5 / 4.7; PREVENT 0.5 → 0.5 | none; COMBO none | 5.4 / 1.9 / 0.3; COVER_9 none | 2.0 / 1.5 / 2.6; BLOWN none | 0.1 / 0.2 / 0.1. — [a04_out.txt](../scratch/audit/a04_out.txt)
- Man share: 35.2 / 37.2 / 34.3 / 30.2 / 28.7% (2018–22), then 41.5 / 48.5 / 31.0% (2023–25). Every coverage type maps one-to-one to a man/zone value (COMBO, COVER_9 and BLOWN map to ZONE). — [a04_out.txt](../scratch/audit/a04_out.txt)
- The NGS taxonomy persists in BDB 2026 for 2023: C3 31.8%, C1 22.5%, C4 16.6%, C2Z 13.0%, C6 9.8%, C0 4.3%, C2M 1.7%, PREVENT 0.3%. These are close to participation 2022 (NGS) and far from participation 2023 (FTN). — [a01_out.txt](../scratch/audit/a01_out.txt), [a04_out.txt](../scratch/audit/a04_out.txt)
- Pass rushers: participation `number_of_pass_rushers` equals FTN `n_pass_rushers` on 99.96% / 99.94% / 100% of plays in 2023 / 2024 / 2025, so it is the FTN field. In 2022 (NGS participation vs FTN) they agree only 76.2% (n = 21,408). Values of 44 and 45 appear (data errors). — [a04_out.txt](../scratch/audit/a04_out.txt)
- FTN charting covers 2022 (41,643 plays), 2023 (48,225), 2024 (48,031), 2025 (47,316) and 2026 through week 5 (13,438). It has no coverage-type column. Blitz rate (n_blitzers > 0) on dropbacks: 23.9% / 27.3% / 28.8% / 29.4% for 2022–25. n_blitzers on dropbacks over all seasons: 0 61,846, 1 17,176, 2 5,221, 3 816, 4+ 63. — [a04_out.txt](../scratch/audit/a04_out.txt)
- Team-season labeled dropbacks by defense (REG): min / median / max of 478 / 538.5 / 632 (2018), 460 / 546 / 663 (2019), 491 / 554 / 668 (2020), 470 / 566.5 / 655 (2021), 477 / 558.5 / 669 (2022), 519 / 645.5 / 734 (2023), 545 / 629 / 713 (2024), 526 / 619.5 / 706 (2025). On 3rd and 4th down, the range is 119–238 per team-season (median 150–197). — [a04_out.txt](../scratch/audit/a04_out.txt)
- Median team-season count by class: Cover 0, 14.5–26.5; Cover 6, 21–44; 2-man, 8–13 (NGS) and 28.5–41 (FTN); Cover 9, 10–16; COMBO, 2–32; PREVENT, 2–3. — [a04_out.txt](../scratch/audit/a04_out.txt)
- BDB 2026 tracking gives 355–520 plays per defensive team in 2023 (median 446.5). — [a07_out.txt](../scratch/audit/a07_out.txt)
- `participation.parquet`: 478,989 rows, no duplicate keys. `old_game_id` is a string, `play_id` a float64, and `defense_man_zone_type` uses '' (69,869 rows in 2023+) as well as null for missing. 636 of 210,954 pbp dropbacks have no participation row. — [a04_out.txt](../scratch/audit/a04_out.txt), [a03_out.txt](../scratch/audit/a03_out.txt)

### Inferences
- Do not pool 2018–22 and 2023+ coverage labels as the same target. The class priors move by up to 14 percentage points, FTN has three classes NGS lacks, and the two sources agree only 57% on the same plays. For tracking models, use the NGS-family labels: BDB 2026 `team_coverage_type`, and nflverse 2018–22 for StatsBomb. FTN labels are best kept for tendency reports, as their own series.
- FTN's own year-to-year swings are large (Cover 1 36.5% → 22.5% and COMBO 5.4% → 0.3% from 2023/24 to 2025), which suggests the charting method changed. Season-over-season tendency comparisons inside the FTN era need a caveat.
- Opponent tendency reports are feasible per team-season for main classes (C1, C3, C2, C4: roughly 60–240 each per team) and for man/zone and blitz rate. They are thin for C0, C6 and 2-man, and especially thin when also split by down (≤ 238 3rd/4th-down dropbacks per team-season). Rare-class tendencies need shrinkage toward league rates or pooling over seasons.

### Gaps
- No local documentation of how NGS and FTN define each coverage. The mapping used here (COVER_N_X ↔ COVER_N, 2-man ↔ COVER_2_MAN, shell groups) is my own.
- Whether 2018–22 participation labels are exactly the same NGS product as BDB's `team_coverage_type` is inferred from matching taxonomy and class mix. No overlapping plays exist to test it, because BDB tracking is 2023-only.

## Overall: data issues, recommended splits, and feasible label targets with class counts

### Takeaway
The cleanest core training set is BDB 2026, 2023 weeks 1–18: 14,105 labeled plays with NGS-taxonomy coverage, man/zone and shell, plus FTN rusher and blitz counts joined on. Split it by week or game. StatsBomb (about 5.2k quality-filtered labeled dropbacks, 2018–22) serves as the pre-snap-motion and out-of-distribution test set, split by game. Per-defender assignments are sparse (3.4k on BDB 2026 plays, 9.5k with BDB 2027 2023 tracking). Blitz cannot be learned from BDB 2026 alignment because the rushers are not tracked.

### Cited Findings
- **ID and type issues:** BDB keys are Int64 `game_id` / `play_id`. Participation has `old_game_id` as a string and `play_id` as a float, so cast both. FTN joins pbp on `nflverse_game_id` + `nflverse_play_id` (the string game id). StatsBomb plays CSV `gsis_play_id` is text like "41.0" (31 null) and `gsis_old_game_id` has 1 null. In StatsBomb tracking JSON, `gsis_play_id` is null on 7 plays, and track `gsis_player_id` is the NFL nfl_id (42 null). BDB 2027 tracking has ISO timestamps instead of frame_id. — [common.py](../scratch/audit/common.py), [a06_out.txt](../scratch/audit/a06_out.txt), [a02_out.txt](../scratch/audit/a02_out.txt)
- **Coordinates and direction:** BDB 2026 and 2027 share raw stadium coordinates (x 0.41–119.86 and y 0.62–52.88 in BDB 2026), with exact x,y equality between them. Neither is normalized for direction (`play_direction` left: 85,832 frame-1 rows; right: 87,318). `absolute_yardline_number` is the LOS in the same x frame: offense median 1.87 / 1.77 yd behind it and defense median 5.0 / 5.1 yd beyond it, for right / left plays. Units: s in yd/s, a in yd/s², `o` and `dir` in degrees, 10 Hz. StatsBomb is 30 Hz, uses its own yard frame (x −14 to 112) with `offense_left_to_right`, and has "NGS" columns that are 6·x + 90 / 6·y + 30. — [a01_out.txt](../scratch/audit/a01_out.txt), [a07_out.txt](../scratch/audit/a07_out.txt)
- **Missing and odd values:** In BDB 2026 supplementary, `team_coverage_type` and man/zone are null on 5 plays (3 tracked), `yardline_side` is null on 251 (midfield), and 3 plays have no Passer. One play has 1 tracked player and 0 defenders, and 6 plays have 11 coverage defenders. Charted rushers include values of 44 and 45. StatsBomb `play_blitz` has no False value and mixed TRUE/True casing, and its depth arrays are unsorted. Participation man/zone uses '' as well as null. — [a01_out.txt](../scratch/audit/a01_out.txt), [a04_out.txt](../scratch/audit/a04_out.txt), [a06_out.txt](../scratch/audit/a06_out.txt)
- **Corrections to DATA.md found during the audit:**
  - "Frame 1 appears to be the snap" is now confirmed.
  - BDB 2027 player_play covers 2023–2025 seasons, 114,954 plays including preseason. It is sparse (1–11 rows per play), not all players.
  - The "14,346 plays with per-player labels" figure counts plays with any row, including 2024. Only 3,163 tracked plays have a labeled coverage defender.
  - StatsBomb gives 6,822 labeled dropbacks when joined through the tracking JSON, not 4,013.
  - The 2021 labeled-dropback count is 18,968, not 19,541.
  - StatsBomb has 124 tracking files.

  — [a03_out.txt](../scratch/audit/a03_out.txt), [a04_out.txt](../scratch/audit/a04_out.txt), [a06_out.txt](../scratch/audit/a06_out.txt)
- **Week structure for splits (BDB 2026):** plays per week range from 666 to 904 (13–16 games). Weeks 1–12 total 9,400 plays, weeks 13–15 total 2,253 and weeks 16–18 total 2,455. The label mix is stable by week: Cover 3 29.4–35.1%, Cover 1 18.6–26.3%, man 25.0–32.4%. — [a07_out.txt](../scratch/audit/a07_out.txt)
- **Feasible targets with class counts on the 14,108 BDB 2026 tracked plays:**
  - Coverage type (NGS): C3 4,482, C1 3,175, C4 2,346, C2Z 1,837, C6 1,388, C0 601, C2M 237, PREVENT 39.
  - Man/zone: 10,092 / 4,013.
  - Shell: single-high 7,657, two-high 5,808, zero 601, PREVENT 39, null 3.
  - FTN rushers: 4 → 9,977, 5 → 2,679, 6 → 698, 3 → 572, 7 → 100, other 78 (0: 46, 1: 3, 2: 19, 8: 9, 44: 1), null 4.
  - FTN blitz (n_blitzers > 0): 3,757 yes / 10,351 no. Number of blitzers: 1 → 2,786, 2 → 848, 3+ → 123.
  - FTN coverage, for reference only: C1 4,483, C3 2,691, C2 2,233, C4 1,689, 2-MAN 835, COMBO 730, C6 729, C0 412, C9 274, null 31, BLOWN 1.

  — [a08_out.txt](../scratch/audit/a08_out.txt), [a01_out.txt](../scratch/audit/a01_out.txt)
- Per-defender targets: 3,426 BDB 2026 defenders have a `coverage_assignment` (MAN 849, deep thirds 665, curl/flat 496, flats 333, quarters 456, deep halves 207, DEEP_FREE 122, hook/curl 227, HOLE 60). BDB 2027 2023 sparse tracking has 9,471 defenders with an assignment and 5,632 with `blitzing=True`. — [a03_out.txt](../scratch/audit/a03_out.txt), [a08_out.txt](../scratch/audit/a08_out.txt)

### Inferences
- **Recommended splits:**
  - BDB 2026: train on weeks 1–12 (9,400), validate on 13–15 (2,253), test on 16–18 (2,455). A GroupKFold by `game_id` over the 272 games works too. Never split plays or frames at random, because frames of one play and plays of one game (same game plan) would leak.
  - BDB 2026 and BDB 2027 2023 tracking overlap on 7,314 plays from the same games, so hold out the same games in both.
  - For an "unseen opponent" estimate, add a leave-teams-out fold, since every team appears every week.
  - StatsBomb: split by game (124) and stratify by whether NE/TB is on defense. Best used as an external test and fine-tuning set: different seasons, a broadcast-derived source and 30 Hz.
- **Target priorities:**
  1. Shell at the snap (3-way; label agreement ceiling about 84% vs FTN, 93% within NGS).
  2. Man/zone (ceiling about 82% vs FTN).
  3. Coverage type with 7 classes, PREVENT dropped or merged.
  4. Per-defender assignment, only as a sparse auxiliary target.
  5. Blitz and rusher count only from StatsBomb (all 22 visible; `play_blitz` and participation rushers for 2017–22) or BDB 2027 sparse tracking, never from BDB 2026 roster membership.
- Feature leakage to guard against:
  - n_coverage and player_role in BDB 2026, which encode the rush.
  - `player_to_predict` and targeted-receiver flags (post-hoc).
  - Any frame after the snap in a "pre-snap" model.
  - Supplementary fields such as `dropback_type`, `pass_location_type` and `route_of_targeted_receiver`, which are outcomes.
- Normalize all sources to an offense-moving-right, LOS-relative frame before modeling: x′ = (x − LOS) × dir_sign, with y mirrored on left plays. Resample StatsBomb from 30 to 10 Hz.

### Gaps
- PFF-vs-NGS charter disagreement for 2022 (BDB 2025) cannot be measured, because BDB 2025 data is no longer available locally.
- I did not test how much the BDB 2026 inclusion bias (rushers missing) affects transfer to a full-22 deployment input such as StatsBomb or broadcast CV. This should be measured by dropping StatsBomb rushers and comparing.
