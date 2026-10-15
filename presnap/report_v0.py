"""Opponent defensive report v0, from charting data only (no tracking).

Run: uv run python -m presnap.report_v0 [TEAM] [SEASON]   (default PHI 2025)

Every rate is shrunk toward the league rate in the same situation (prior weight K plays), so small
samples can't produce loud tendencies. Shell here is the shell of the coverage PLAYED; the pre-snap
shell shown and rotation need tracking (Phase 1).
"""
import sys
from pathlib import Path

import polars as pl

from presnap import io, labels

K = 20          # shrinkage prior weight, in plays
MIN_N = 20      # below this a row is flagged as thin
OUT = Path(__file__).resolve().parents[1] / "reports"
FAMS = ["C1", "C3", "C2", "QTR", "C0", "C2M"]


def load(season):
    p = io.pbp(["old_game_id", "play_id", "season", "season_type", "week", "defteam", "posteam", "down", "ydstogo",
                "yardline_100", "qb_dropback", "wp", "qtr"]).filter(
        (pl.col("season") == season) & (pl.col("season_type") == "REG") & (pl.col("qb_dropback") == 1)
        & pl.col("wp").is_between(0.05, 0.95))  # drop garbage time
    part = io.participation().select("old_game_id", "play_id", "offense_personnel", "offense_formation",
                                     coverage=labels.canon("defense_coverage_type"))
    f = io.ftn().select("old_game_id", "play_id", "n_blitzers", "n_pass_rushers", "is_motion")
    d = p.join(part, on=["old_game_id", "play_id"]).join(f, on=["old_game_id", "play_id"], how="left")
    d = labels.with_label_columns(d).filter(pl.col("coverage").is_not_null())
    return d.with_columns(
        man=(pl.col("man_zone") == "MAN").cast(pl.Float64),
        two_high=pl.when(pl.col("shell_played").is_not_null()).then((pl.col("shell_played") == 2).cast(pl.Float64)),
        blitz=pl.when(pl.col("n_blitzers").is_not_null()).then((pl.col("n_blitzers") > 0).cast(pl.Float64)),
        dd=pl.when(pl.col("down") == 1).then(pl.lit("1st down"))
        .when(pl.col("down") == 2).then(pl.when(pl.col("ydstogo") <= 3).then(pl.lit("2nd & 1-3")).when(pl.col("ydstogo") <= 6).then(pl.lit("2nd & 4-6")).otherwise(pl.lit("2nd & 7+")))
        .when(pl.col("down") == 3).then(pl.when(pl.col("ydstogo") <= 3).then(pl.lit("3rd & 1-3")).when(pl.col("ydstogo") <= 6).then(pl.lit("3rd & 4-6")).when(pl.col("ydstogo") <= 10).then(pl.lit("3rd & 7-10")).otherwise(pl.lit("3rd & 11+")))
        .otherwise(pl.lit("4th down")),
        zone=pl.when(pl.col("yardline_100") >= 80).then(pl.lit("Backed up (own 1-20)"))
        .when(pl.col("yardline_100") > 50).then(pl.lit("Own 21-49"))
        .when(pl.col("yardline_100") > 20).then(pl.lit("Opp 50-21"))
        .when(pl.col("yardline_100") > 10).then(pl.lit("Red zone (20-11)"))
        .otherwise(pl.lit("Tight red zone (10-1)")),
        personnel=pl.col("offense_personnel").str.extract(r"(\d) RB").fill_null("0") + pl.col("offense_personnel").str.extract(r"(\d) TE").fill_null("0"),
        motion=pl.when(pl.col("is_motion")).then(pl.lit("Motion")).otherwise(pl.lit("No motion")),
    )


def shrink(k, n, prior):
    return (k + K * prior) / (n + K)


def table(team, league, by, order=None):
    lg = league.group_by(by).agg(lg_man=pl.col("man").mean(), lg_blitz=pl.col("blitz").mean(), lg_two=pl.col("two_high").mean())
    t = team.group_by(by).agg(
        n=pl.len(), man_k=pl.col("man").sum(), blitz_k=pl.col("blitz").sum(), blitz_n=pl.col("blitz").count(),
        two_k=pl.col("two_high").sum(), two_n=pl.col("two_high").count(),
        **{f: (pl.col("family") == f).mean() for f in FAMS},
    ).join(lg, on=by, how="left")
    t = t.with_columns(
        man=shrink(pl.col("man_k"), pl.col("n"), pl.col("lg_man")),
        blitz=shrink(pl.col("blitz_k"), pl.col("blitz_n"), pl.col("lg_blitz")),
        two_high=shrink(pl.col("two_k"), pl.col("two_n"), pl.col("lg_two")),
    )
    if order:
        t = t.with_columns(pl.col(by).cast(pl.Enum(order))).sort(by).with_columns(pl.col(by).cast(pl.Utf8))
    else:
        t = t.sort("n", descending=True)
    return t


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def md(t, by, label):
    head = f"| {label} | n | Man | Lg man | 2-high | Blitz | Lg blitz | " + " | ".join(FAMS) + " |"
    rows = [head, "|" + "---|" * (7 + len(FAMS))]
    for r in t.iter_rows(named=True):
        thin = " †" if r["n"] < MIN_N else ""
        rows.append(f"| {r[by]}{thin} | {r['n']} | **{pct(r['man'])}** | {pct(r['lg_man'])} | {pct(r['two_high'])} | **{pct(r['blitz'])}** | {pct(r['lg_blitz'])} | "
                    + " | ".join(pct(r[f]) for f in FAMS) + " |")
    return "\n".join(rows)


def tendencies(team, league, base):
    """Situations where the shrunk rate moves furthest from the team's own baseline (n >= 25)."""
    out = []
    for by in ("dd", "zone", "personnel", "motion"):
        t = table(team, league, by).filter(pl.col("n") >= 25)
        for r in t.iter_rows(named=True):
            for m, lab in (("man", "man coverage"), ("blitz", "blitz"), ("two_high", "two-high coverage")):
                if r[m] is not None and base[m] is not None:
                    out.append((abs(r[m] - base[m]), f"**{r[by]}**: {lab} {pct(r[m])} vs {pct(base[m])} overall (n={r['n']})"))
    return [s for _, s in sorted(out, reverse=True)[:5]]


def main(team="PHI", season=2025):
    d = load(season)
    league, me = d, d.filter(pl.col("defteam") == team)
    base = {m: shrink(me[m].sum(), me[m].count(), league[m].mean()) for m in ("man", "blitz", "two_high")}
    mix = me["family"].value_counts(normalize=True).join(league["family"].value_counts(normalize=True), on="family", suffix="_lg", how="left")
    mix_rows = "\n".join(f"| {r['family'] or 'other (combo/blown)'} | {pct(r['proportion'])} | {pct(r['proportion_lg'])} |"
                         for r in mix.sort("proportion", descending=True).iter_rows(named=True))
    games = me["old_game_id"].n_unique()
    sections = [
        f"# {team} defense, {season} regular season: pre-game report v0",
        f"*Charting-only baseline: FTN coverage labels via nflverse. {me.height} dropbacks in {games} games, garbage time excluded (win probability outside 5–95%). "
        f"Rates are shrunk toward the league rate for the same situation (prior = {K} plays). † = fewer than {MIN_N} plays.*",
        "## Snapshot",
        f"| | {team} | League |\n|---|---|---|\n| Man coverage | **{pct(base['man'])}** | {pct(league['man'].mean())} |\n"
        f"| Two-high (played) | **{pct(base['two_high'])}** | {pct(league['two_high'].mean())} |\n"
        f"| Blitz (FTN: 1+ blitzer) | **{pct(base['blitz'])}** | {pct(league['blitz'].mean())} |\n"
        f"| Avg pass rushers | {me['n_pass_rushers'].mean():.2f} | {league['n_pass_rushers'].mean():.2f} |",
        "### Coverage mix\n| Coverage | " + team + " | League |\n|---|---|---|\n" + mix_rows,
        "## Top tendencies\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(tendencies(me, league, base), 1)),
        "## By down and distance\n" + md(table(me, league, "dd", ["1st down", "2nd & 1-3", "2nd & 4-6", "2nd & 7+", "3rd & 1-3", "3rd & 4-6", "3rd & 7-10", "3rd & 11+", "4th down"]), "dd", "Situation"),
        "## By field zone\n" + md(table(me, league, "zone", ["Backed up (own 1-20)", "Own 21-49", "Opp 50-21", "Red zone (20-11)", "Tight red zone (10-1)"]), "zone", "Field zone"),
        "## By offensive personnel (RB, TE)\n" + md(table(me, league, "personnel"), "personnel", "Personnel"),
        "## Against motion\n" + md(table(me, league, "motion", ["Motion", "No motion"]), "motion", "Pre-snap"),
        "## What v0 can't tell you yet\n"
        "- **What they show before the snap.** These shells are the coverage *played*. Shell at the snap, rotation and disguise rate need tracking (Phase 1).\n"
        "- **Label noise.** FTN and NGS agree on the exact coverage only about 57% of the time on shared plays (shell about 84%, man/zone about 82%). Treat family-level splits as approximate.\n"
        "- **Clips.** No video links yet. Every tendency should eventually link to its plays.\n"
        "- Data: FTN Data via nflverse (CC-BY-SA 4.0); nflverse play-by-play.",
    ]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{team.lower()}_{season}_defense_v0.md"
    path.write_text("\n\n".join(sections) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["PHI"]), *([int(sys.argv[2])] if len(sys.argv) > 2 else []))
