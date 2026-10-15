"""Phase 1A human review: 50 held-out plays drawn as top-down diagrams (QB view) at the snap and at +2.0 s.
36 are plays where the rule at +2.0 s disagrees with the NGS played shell, 14 are random agreements
(blind control). Labels and rule calls are hidden. Answers save in the browser and download as CSV.

Run (on the PC): uv run python -m presnap.review   -> data/review/shell_review.html + key.csv
The page holds BDB-derived positions, so it lives under data/ (not committed).
"""
import json

import polars as pl

from presnap import io
from presnap.shell import KEY, POST_FRAME

OUT = io.ROOT / "data/review"
S = 9            # px per yard
X0, X1 = -8, 26  # depth window (yards past LOS)
W = 53.3
DISPUTES = {(2, 1): 12, (1, 2): 12, (0, 1): 4, (1, 0): 4, (0, 2): 2, (2, 0): 2}
N_AGREE = 14


def pick(plays, shell):
    d = shell.join(plays.select(KEY + ["shell_played", "split", "down", "ydstogo", "offense_formation", "receiver_alignment"]), on=KEY)
    d = d.filter((pl.col("source") == "bdb26") & (pl.col("split") != "train") & pl.col("shell_played").is_not_null()
                 & pl.col("shell_throw").is_not_null())
    parts = [d.filter((pl.col("shell_played") == a) & (pl.col("shell_throw") == b)).sample(n, seed=1) for (a, b), n in DISPUTES.items()]
    parts.append(d.filter(pl.col("shell_played") == pl.col("shell_throw")).sample(N_AGREE, seed=1))
    return pl.concat(parts).sample(fraction=1, shuffle=True, seed=2).with_row_index("i").with_columns(
        review_id=pl.format("R{}", (pl.col("i") + 1).cast(pl.Utf8).str.zfill(2)))


def svg(rows):
    w, h = W * S, (X1 - X0) * S
    px = lambda y: (W / 2 - y) * S          # + y_rel = offense's left = screen left
    py = lambda x: (X1 - x) * S             # downfield is up
    g = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" role="img">',
         f'<rect width="{w:.0f}" height="{h:.0f}" class="turf"/>']
    for x in range(X0 - X0 % 5, X1 + 1, 5):
        g.append(f'<line x1="0" x2="{w:.0f}" y1="{py(x):.0f}" y2="{py(x):.0f}" class="yl"/>'
                 f'<text x="4" y="{py(x) - 3:.0f}" class="yd">{x:+d}</text>')
    for y in (3.08, -3.08):
        g.append(f'<line x1="{px(y):.0f}" x2="{px(y):.0f}" y1="0" y2="{h:.0f}" class="hash"/>')
    for y in (14.65, -14.65):
        g.append(f'<line x1="{px(y):.0f}" x2="{px(y):.0f}" y1="0" y2="{h:.0f}" class="num"/>')
    g.append(f'<line x1="0" x2="{w:.0f}" y1="{py(0):.0f}" y2="{py(0):.0f}" class="los"/>')
    for r in rows:
        cx, cy = px(r["y_rel"]), py(r["x_rel"])
        if r["side"] == "def":
            if r["vx"] is not None:
                g.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{px(r["y_rel"] + r["vy"] * .5):.1f}" y2="{py(r["x_rel"] + r["vx"] * .5):.1f}" class="vel"/>')
            g.append(f'<rect x="{cx - 6:.1f}" y="{cy - 6:.1f}" width="12" height="12" class="def"/>')
        else:
            g.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" class="off"/>')
        g.append(f'<text x="{cx:.1f}" y="{cy - 9:.1f}" class="lbl">{r["position"] or ""}</text>')
    g.append("</svg>")
    return "".join(g)


CSS = """
:root{--bg:#f6f7f5;--ink:#1d2321;--muted:#5b6461;--card:#fff;--line:#d7dcd9;--turf:#e8f1ea;--def:#c0392b;--off:#3d5a80;--accent:#1f7a4d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121614;--ink:#e7ece9;--muted:#9aa6a1;--card:#1b211e;--line:#2f3833;--turf:#18241c;--def:#e57368;--off:#8fb0d9;--accent:#4cc38a}}
:root[data-theme=dark]{--bg:#121614;--ink:#e7ece9;--muted:#9aa6a1;--card:#1b211e;--line:#2f3833;--turf:#18241c;--def:#e57368;--off:#8fb0d9;--accent:#4cc38a}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:16px 0}
.pair{display:flex;gap:12px;flex-wrap:wrap}.pair figure{margin:0;flex:1 1 300px}.pair svg{width:100%;height:auto}
figcaption{font-weight:600;margin-bottom:4px}
.turf{fill:var(--turf)}.yl{stroke:var(--line)}.hash{stroke:var(--muted);stroke-dasharray:3 4;opacity:.6}.num{stroke:var(--muted);opacity:.25}
.los{stroke:var(--accent);stroke-width:2}.yd{fill:var(--muted);font-size:10px}
.def{fill:var(--def)}.off{fill:var(--off)}.vel{stroke:var(--def);stroke-width:2;opacity:.7}
.lbl{fill:var(--ink);font-size:9px;text-anchor:middle}
fieldset{border:0;padding:0;margin:8px 0}legend{font-weight:600}label{margin-right:14px;white-space:nowrap}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-weight:600;cursor:pointer}
input[type=text]{width:100%;box-sizing:border-box;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
"""

JS = """
const KEY='presnap-shell-review-v1';let A={};try{A=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(A))}catch(e){}count()}
function count(){const n=IDS.filter(i=>A[i]&&A[i].snap&&A[i].post).length;document.getElementById('done').textContent=n+' / '+IDS.length+' done'}
document.querySelectorAll('input[type=radio]').forEach(r=>{const[id,f]=r.name.split('|');if(A[id]&&A[id][f]===r.value)r.checked=true;
 r.addEventListener('change',()=>{A[id]=A[id]||{};A[id][f]=r.value;save()})});
document.querySelectorAll('input[type=text]').forEach(t=>{const id=t.name;if(A[id]&&A[id].note)t.value=A[id].note;
 t.addEventListener('input',()=>{A[id]=A[id]||{};A[id].note=t.value;save()})});
document.getElementById('dl').onclick=()=>{const rows=[['review_id','snap','post','note']].concat(IDS.map(i=>[i,(A[i]||{}).snap||'',(A[i]||{}).post||'',JSON.stringify((A[i]||{}).note||'')]));
 const b=new Blob([rows.map(r=>r.join(',')).join('\\n')],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='shell_review_answers.csv';a.click()};
count();
"""


def main():
    plays = pl.read_parquet(io.PROCESSED / "plays.parquet")
    shell = pl.read_parquet(io.PROCESSED / "shell.parquet")
    sel = pick(plays, shell)
    t = pl.scan_parquet(io.PROCESSED / "tracking.parquet").join(sel.lazy().select(KEY), on=KEY).collect()
    last = t.group_by(KEY).agg(pl.col("frame").max().alias("last"))
    t = t.join(last, on=KEY)
    cards = []
    for r in sel.iter_rows(named=True):
        p = t.filter((pl.col("game_id") == r["game_id"]) & (pl.col("play_id") == r["play_id"]))
        post = min(POST_FRAME, p["last"][0])
        a = svg(p.filter(pl.col("frame") == 0).to_dicts())
        b = svg(p.filter(pl.col("frame") == post).to_dicts())
        rid = r["review_id"]
        radios = lambda f: "".join(f'<label><input type="radio" name="{rid}|{f}" value="{v}"> {v}</label>' for v in ("0-high", "1-high", "2-high", "unsure"))
        cards.append(f'<section class="card"><h2>{rid}</h2><p class="muted">{int(r["down"])} & {int(r["ydstogo"])} · {r["offense_formation"] or ""} {r["receiver_alignment"] or ""}</p>'
                     f'<div class="pair"><figure><figcaption>At the snap</figcaption>{a}</figure><figure><figcaption>+{post / 10:.1f} s</figcaption>{b}</figure></div>'
                     f'<fieldset><legend>Shell at the snap</legend>{radios("snap")}</fieldset>'
                     f'<fieldset><legend>Shell at +{post / 10:.1f} s</legend>{radios("post")}</fieldset>'
                     f'<input type="text" name="{rid}" placeholder="Optional note"></section>')
    ids = sel["review_id"].to_list()
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Shell review</title><style>{CSS}</style></head><body><main>'
            '<h1>Shell review: 50 plays</h1>'
            '<p class="muted">QB view: offense (blue circles) at the bottom, defense (red squares) above, green line = line of scrimmage, dashed = hashes, faint = numbers. '
            'Red ticks show where each defender is heading over the next 0.5 s. Pass rushers and linemen aren\'t tracked in this data, so only coverage defenders appear. '
            'For each play, call the shell at the snap and after rotation. Use "unsure" freely. Answers save in this browser.</p>'
            '<div class="bar"><span id="done"></span><button id="dl">Download answers (CSV)</button></div>'
            + "".join(cards) + f'</main><script>const IDS={json.dumps(ids)};{JS}</script></body></html>')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shell_review.html").write_text(html)
    sel.drop("i").write_csv(OUT / "key.csv")
    print("wrote", OUT / "shell_review.html", len(ids), "plays")


if __name__ == "__main__":
    main()
