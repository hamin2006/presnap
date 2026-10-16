"""Phase 3: QB pre-game report v1 for one defense (2023 season, tracking + NGS labels).

  uv run python -m presnap.report [TEAM]   ->  reports/<team>_2023_defense_v1.html   (default PHI)

Sources: BDB 2026 tracking (snap -> throw, 2023) with NGS coverage labels; the Phase 1A shell rule at the snap and at
+2.0 s (provisional until the 1A review); out-of-fold model predictions (presnap.oof); FTN blitz/motion charting.
Every situational rate is shrunk toward the league rate in the same situation (prior = K plays).
"""
import html
import sys
from pathlib import Path

import polars as pl

from presnap import features, io
from presnap.report_v0 import shrink, K
from presnap.shell import KEY

OUT = Path(__file__).resolve().parents[1] / "reports"
FAMS = ["C1", "C3", "C2", "QTR", "C0", "C2M"]
FAM_NAMES = {"C0": "Cover 0", "C1": "Cover 1", "C2": "Cover 2", "C3": "Cover 3", "QTR": "Quarters (4/6)", "C2M": "2-Man"}
SHELL = {0: "0-high", 1: "1-high", 2: "2-high"}
COV7 = ["C0", "C1", "C2M", "C2", "C3", "C4", "C6"]
TO_FAM = {"C0": "C0", "C1": "C1", "C2M": "C2M", "C2": "C2", "C3": "C3", "C4": "QTR", "C6": "QTR"}


def load():
    p = pl.read_parquet(io.PROCESSED / "plays.parquet").filter((pl.col("source") == "bdb26") & pl.col("family").is_not_null())
    sh = pl.read_parquet(io.PROCESSED / "shell.parquet").select(*KEY, "shell_snap", "shell_throw")
    oof = pl.read_parquet(io.PROCESSED / "oof_predictions.parquet").with_columns(source=pl.lit("bdb26"))
    snap = features.build("bdb26", {"snap": 0}).select(*KEY, "snap_cb1_cushion", "snap_cb2_cushion", "snap_x1", "snap_x2")
    d = p.join(sh, on=KEY, how="left").join(oof, on=KEY, how="left").join(snap, on=KEY, how="left")
    fam_of = lambda c: pl.col(c).replace_strict({i: TO_FAM[x] for i, x in enumerate(COV7)}, default=None)
    return d.with_columns(
        man=(pl.col("man_zone") == "MAN").cast(pl.Float64),
        two_played=(pl.col("shell_played") == 2).cast(pl.Float64),
        two_shown=(pl.col("shell_snap") == 2).cast(pl.Float64),
        rotated=(pl.col("shell_snap") != pl.col("shell_throw")).cast(pl.Float64),
        blitz=pl.when(pl.col("ftn_n_blitzers").is_not_null()).then((pl.col("ftn_n_blitzers") > 0).cast(pl.Float64)),
        read_snap=(fam_of("cov_snap") == pl.col("family")).cast(pl.Float64),
        read_1s=(fam_of("cov_t10") == pl.col("family")).cast(pl.Float64),
        press=pl.min_horizontal("snap_cb1_cushion", "snap_cb2_cushion") < 2.0,
        dd=pl.when(pl.col("down") == 1).then(pl.lit("1st down"))
        .when(pl.col("down") == 2).then(pl.when(pl.col("ydstogo") <= 3).then(pl.lit("2nd & short (1-3)")).when(pl.col("ydstogo") <= 6).then(pl.lit("2nd & medium (4-6)")).otherwise(pl.lit("2nd & long (7+)")))
        .when(pl.col("down") >= 3).then(pl.when(pl.col("ydstogo") <= 3).then(pl.lit("3rd/4th & short (1-3)")).when(pl.col("ydstogo") <= 6).then(pl.lit("3rd/4th & medium (4-6)")).otherwise(pl.lit("3rd/4th & long (7+)"))),
        zone=pl.when(pl.col("yardline_100") > 20).then(pl.lit("Open field")).when(pl.col("yardline_100") > 10).then(pl.lit("Red zone (20-11)")).otherwise(pl.lit("Goal-to-go area (10-1)")),
    )


def pct(x, digits=0):
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def rates(team, league, by, measures):
    lg = league.group_by(by).agg([pl.col(m).mean().alias(f"lg_{m}") for m in measures])
    t = team.group_by(by).agg([pl.len().alias("n")] + [pl.col(m).sum().alias(f"{m}_k") for m in measures]
                              + [pl.col(m).count().alias(f"{m}_n") for m in measures]
                              + [(pl.col("family") == f).mean().alias(f) for f in FAMS]).join(lg, on=by, how="left")
    return t.with_columns([shrink(pl.col(f"{m}_k"), pl.col(f"{m}_n"), pl.col(f"lg_{m}")).alias(m) for m in measures])


def table(headers, rows, cls=""):
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def bar(x):
    return f'<span class="bar"><span style="width:{0 if x is None else 100 * x:.0f}%"></span></span> {pct(x)}'


CSS = """
:root{--bg:#f5f6f4;--card:#fff;--ink:#1b211f;--muted:#5c6662;--line:#d9dedb;--accent:#1f7a4d;--warn:#b5651d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111513;--card:#1a201d;--ink:#e6ebe8;--muted:#9aa5a0;--line:#2e3732;--accent:#4cc38a;--warn:#e0a060}}
:root[data-theme=dark]{--bg:#111513;--card:#1a201d;--ink:#e6ebe8;--muted:#9aa5a0;--line:#2e3732;--accent:#4cc38a;--warn:#e0a060}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:24px;margin:8px 0 4px}h2{font-size:18px;margin:28px 0 8px}.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:12px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.card b{font-size:22px;display:block}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{font-size:13px;color:var(--muted)}
.bar{display:inline-block;width:60px;height:8px;background:var(--line);border-radius:4px;vertical-align:middle;overflow:hidden}.bar span{display:block;height:100%;background:var(--accent)}
ol li{margin:4px 0}.note{border-left:3px solid var(--warn);padding:6px 10px;background:var(--card);margin:10px 0}
"""


def main(team="PHI"):
    d = load()
    league, me = d, d.filter(pl.col("defense") == team)
    m = ["man", "two_played", "two_shown", "rotated", "blitz", "read_snap", "read_1s"]
    base = {k: me[k].mean() for k in m}
    lg = {k: league[k].mean() for k in m}
    teams = league.group_by("defense").agg(pl.col("read_snap").mean(), pl.col("read_1s").mean(), pl.col("rotated").mean(), pl.len()).with_columns(
        hidden=pl.col("read_1s") - pl.col("read_snap")).sort("read_snap")
    rank = teams["defense"].to_list().index(team) + 1
    sections = []
    sections.append(f'<h1>{team} defense: QB pre-game report</h1><p class="muted">2023 regular season, {me.height} dropbacks with tracking. '
                    f'Coverage labels: NGS. Rates in tables are pulled toward the league rate for the same situation (prior = {K} plays).</p>')
    cards = [("Man coverage", base["man"], lg["man"]), ("Two-high shown at snap", base["two_shown"], lg["two_shown"]),
             ("Two-high played", base["two_played"], lg["two_played"]), ("Shell changes after snap", base["rotated"], lg["rotated"]),
             ("Blitz (FTN)", base["blitz"], lg["blitz"])]
    sections.append('<div class="cards">' + "".join(f'<div class="card"><span class="muted">{n}</span><b>{pct(v)}</b><span class="muted">league {pct(l)}</span></div>' for n, v, l in cards) + "</div>")

    sections.append("<h2>How readable are they?</h2>")
    sections.append(f"<p>Our model, which never saw these plays in training, names the coverage family correctly <b>{pct(base['read_snap'])}</b> of the time at the snap "
                    f"(league {pct(lg['read_snap'])}) and <b>{pct(base['read_1s'])}</b> one second later (league {pct(lg['read_1s'])}). "
                    f"Ranked <b>{rank} of {teams.height}</b> on readability at the snap (1 = hardest to read). The gap is what they hide until after the snap.</p>")

    t = me.drop_nulls("shell_snap").group_by("shell_snap").agg([pl.len().alias("n")] + [(pl.col("family") == f).mean().alias(f) for f in FAMS]).sort("shell_snap")
    sections.append("<h2>What they show vs. what they play</h2><p class=\"muted\">Rows: the look at the snap (safety shell, rule-based). Columns: coverage actually played.</p>")
    two = t.filter(pl.col("shell_snap") == 2)
    if two.height:
        r2 = two.row(0, named=True)
        sections.append(f'<p><b>Key read:</b> when they show two-high at the snap ({r2["n"]} plays), it ends up single-high '
                        f'(Cover 1 or Cover 3) <b>{pct(r2["C1"] + r2["C3"])}</b> of the time and stays two-high (Cover 2, Quarters or 2-Man) '
                        f'{pct(r2["C2"] + r2["QTR"] + r2["C2M"])}.</p>')
    sections.append(table(["Look at the snap", "Plays"] + [FAM_NAMES[f] for f in FAMS],
                          [[SHELL[r["shell_snap"]], r["n"]] + [bar(r[f]) for f in FAMS] for r in t.iter_rows(named=True)]))

    order = ["1st down", "2nd & short (1-3)", "2nd & medium (4-6)", "2nd & long (7+)", "3rd/4th & short (1-3)", "3rd/4th & medium (4-6)", "3rd/4th & long (7+)"]
    s = rates(me, league, "dd", ["man", "two_shown", "two_played", "rotated", "blitz"]).with_columns(pl.col("dd").cast(pl.Enum(order))).sort("dd")
    sections.append("<h2>By down and distance</h2>")
    sections.append(table(["Situation", "Plays", "Man", "2-high shown", "2-high played", "Shell changes", "Blitz", "Most common"],
                          [[r["dd"], r["n"] if r["n"] >= 20 else f'{r["n"]} †', pct(r["man"]), pct(r["two_shown"]), pct(r["two_played"]), pct(r["rotated"]), pct(r["blitz"]),
                            FAM_NAMES[max(FAMS, key=lambda f: r[f] or 0)]] for r in s.iter_rows(named=True)]))
    z = rates(me, league, "zone", ["man", "two_played", "rotated", "blitz"])
    sections.append("<h2>By field zone</h2>")
    sections.append(table(["Field zone", "Plays", "Man", "2-high played", "Shell changes", "Blitz"],
                          [[r["zone"], r["n"], pct(r["man"]), pct(r["two_played"]), pct(r["rotated"]), pct(r["blitz"])] for r in z.sort("n", descending=True).iter_rows(named=True)]))

    cues = []
    for name, cond in (("Corner pressed (< 2 yd cushion)", pl.col("press")), ("Corners off (2+ yd)", ~pl.col("press")),
                       ("Pre-snap motion", pl.col("ftn_is_motion") == True), ("No motion", pl.col("ftn_is_motion") == False),
                       ("Deepest safety 14+ yd", pl.col("snap_x1") >= 14), ("Deepest safety < 11 yd", pl.col("snap_x1") < 11)):
        a, b = me.filter(cond), league.filter(cond)
        if a.height:
            cues.append([name, a.height, pct(shrink(a["man"].sum(), a.height, b["man"].mean())), pct(b["man"].mean()),
                         pct(shrink(a["two_played"].sum(), a.height, b["two_played"].mean())), pct(b["two_played"].mean())])
    sections.append("<h2>Pre-snap cues</h2>")
    sections.append(table(["Cue at the snap", "Plays", "Man", "League man", "2-high played", "League 2-high"], cues))

    tells = []
    for by in ("dd", "zone"):
        r_ = rates(me, league, by, ["man", "two_played", "rotated", "blitz"]).filter(pl.col("n") >= 25)
        for r in r_.iter_rows(named=True):
            for k, lab in (("man", "man coverage"), ("two_played", "two-high coverage"), ("rotated", "a shell change after the snap"), ("blitz", "a blitz")):
                if r[k] is not None and base[k] is not None:
                    tells.append((abs(r[k] - base[k]), f"<b>{r[by]}</b>: {lab} {pct(r[k])} vs {pct(base[k])} overall ({r['n']} plays)"))
    sections.insert(2, "<h2>Top tendencies</h2><ol>" + "".join(f"<li>{s_}</li>" for _, s_ in sorted(tells, reverse=True)[:6]) + "</ol>")

    sections.append('<div class="note"><b>Read with care.</b> The shell at the snap comes from a geometry rule that agrees with NGS 82% of the time after the snap; '
                    'its human review is still pending. NGS and FTN charting agree on the exact coverage only ~57% of the time, so treat family splits as approximate. '
                    '† = fewer than 20 plays. Pass rushers aren\'t tracked in this data, so blitz numbers come from FTN charting.</div>')
    sections.append('<p class="muted">Data: NFL Big Data Bowl 2026 (NGS tracking and labels), FTN Data via nflverse (CC-BY-SA 4.0). Built with presnap.</p>')
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{team} Defense Report</title><style>{CSS}</style></head><body><main>' + "".join(sections) + "</main></body></html>")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{team.lower()}_2023_defense_v1.html"
    path.write_text(page)
    print("wrote", path)


if __name__ == "__main__":
    main(*sys.argv[1:])
