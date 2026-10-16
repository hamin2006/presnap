"""Write the local click tool (data/review/click/index.html) for the frames in data/review/click/frames/.
Ground truth for Phase 2: the 22 players' feet plus 4 field landmarks per snap frame.
Run: uv run python -m presnap.video.click_tool"""
import json

from presnap import io

D = io.ROOT / "data/review/click"
LINES = (["Left goal line"] + [f"Left {k}" for k in range(5, 50, 5)] + ["50"]
         + [f"Right {k}" for k in range(45, 0, -5)] + ["Right goal line"])

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Snap Click Labels</title><style>
:root{--bg:#f4f5f3;--panel:#fff;--ink:#1c2220;--muted:#5d6763;--line:#d6dbd8;--accent:#1f7a4d;--pl:#d0368a;--lm:#0a84c6}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111513;--panel:#1a201d;--ink:#e6ebe8;--muted:#9aa5a0;--line:#2e3732;--accent:#4cc38a;--pl:#ff6fb5;--lm:#4fc3ff}}
:root[data-theme=dark]{--bg:#111513;--panel:#1a201d;--ink:#e6ebe8;--muted:#9aa5a0;--line:#2e3732;--accent:#4cc38a;--pl:#ff6fb5;--lm:#4fc3ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0 8px 0 0}.muted{color:var(--muted)}
button,select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font:inherit;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}button.on{outline:2px solid var(--accent)}
main{flex:1;display:flex;min-height:0}#wrap{flex:1;position:relative;min-width:0}canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;background:#000}
aside{width:300px;border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:12px 14px}
aside ol{padding-left:18px;margin:6px 0}aside li{cursor:pointer;margin:2px 0}aside li.cur{font-weight:700;color:var(--accent)}
.ok{color:var(--accent)}#lmform{position:absolute;display:none;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;z-index:5;box-shadow:0 4px 16px #0006}
#lmform label{display:block;margin:6px 0 2px}kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
@media (max-width:800px){main{flex-direction:column}aside{width:auto;max-height:40vh;border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<header><h1>Snap click labels</h1>
<button id="mP" class="on">Players (P)</button><button id="mL">Landmarks (L)</button>
<button id="prev">◀ Prev (B)</button><span id="title" class="muted"></span><button id="next">Next (N) ▶</button>
<button id="fit">Fit (F)</button><button id="undo">Undo (Z)</button><button id="dl" class="primary">Download labels</button>
<label class="muted">Restore <input type="file" id="imp" accept=".json"></label></header>
<main><div id="wrap"><canvas id="cv"></canvas>
<div id="lmform"><b>Which landmark?</b><label>Yard line (as seen in the image)</label><select id="lmline"></select>
<label>Where it meets</label><select id="lmedge"><option value="near">near sideline (bottom of image)</option><option value="far">far sideline (top of image)</option></select>
<div style="margin-top:8px;display:flex;gap:6px"><button id="lmok" class="primary">Add</button><button id="lmcancel">Cancel</button></div></div></div>
<aside><p><b>Per frame:</b> click the <span style="color:var(--pl)">feet of all 22 players</span> on the field (between their feet; skip officials and sideline people), then <span style="color:var(--lm)">4 landmarks</span>: pick two clearly visible yard lines and click where each meets the near and the far sideline (the field-side edge of the white border).</p>
<p class="muted">Scroll to zoom, drag to pan. <kbd>Shift</kbd>+click removes the nearest point. Saves automatically in this browser; download when done and put the file in <code>presnap/data/review/</code>.</p>
<div id="counts"></div><ol id="list"></ol></aside></main>
<script>
const IDS = __IDS__, LINES = __LINES__, KEY = 'presnap-click-v1';
let S = {}; try { S = JSON.parse(localStorage.getItem(KEY) || '{}') } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)) } catch (e) {} render(); };
let cur = 0, mode = 'P', img = new Image(), sc = 1, ox = 0, oy = 0, pending = null;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d'), form = document.getElementById('lmform');
LINES.forEach(l => { const o = document.createElement('option'); o.textContent = l; document.getElementById('lmline').appendChild(o); });
const rec = () => (S[IDS[cur]] = S[IDS[cur]] || { players: [], landmarks: [] });
function fit() { const r = cv.getBoundingClientRect(); sc = Math.min(r.width / img.width, r.height / img.height); ox = (r.width - img.width * sc) / 2; oy = (r.height - img.height * sc) / 2; draw(); }
function load(i) { cur = (i + IDS.length) % IDS.length; img = new Image(); img.onload = fit; img.src = 'frames/' + IDS[cur] + '.jpg'; render(); }
function resize() { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d; ctx.setTransform(d, 0, 0, d, 0, 0); draw(); }
function draw() {
  const r = cv.getBoundingClientRect(); ctx.clearRect(0, 0, r.width, r.height);
  if (!img.complete) return; ctx.drawImage(img, ox, oy, img.width * sc, img.height * sc);
  const R = rec(), st = getComputedStyle(document.documentElement), rad = Math.max(3, Math.min(8, 6 * sc));
  ctx.lineWidth = 2;
  ctx.strokeStyle = st.getPropertyValue('--pl'); R.players.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x * sc + ox, y * sc + oy, rad, 0, 7); ctx.stroke(); });
  ctx.strokeStyle = st.getPropertyValue('--lm'); ctx.fillStyle = st.getPropertyValue('--lm'); ctx.font = '12px system-ui';
  R.landmarks.forEach(m => { const X = m.x * sc + ox, Y = m.y * sc + oy; ctx.strokeRect(X - rad, Y - rad, 2 * rad, 2 * rad); ctx.fillText(m.line + ' / ' + m.edge, X + rad + 3, Y - rad); });
}
function render() {
  const R = rec(); document.getElementById('title').textContent = (cur + 1) + ' / ' + IDS.length;
  document.getElementById('counts').innerHTML = 'This frame: <b class="' + (R.players.length === 22 ? 'ok' : '') + '">' + R.players.length + '/22 players</b>, <b class="' + (R.landmarks.length >= 4 ? 'ok' : '') + '">' + R.landmarks.length + '/4 landmarks</b>';
  const L = document.getElementById('list'); L.innerHTML = '';
  IDS.forEach((id, i) => { const r = S[id] || { players: [], landmarks: [] }, li = document.createElement('li');
    li.textContent = 'Play ' + (i + 1) + ': ' + r.players.length + ' players, ' + r.landmarks.length + ' landmarks' + (r.players.length >= 20 && r.landmarks.length >= 4 ? ' ✓' : '');
    if (i === cur) li.className = 'cur'; li.onclick = () => load(i); L.appendChild(li); });
  draw();
}
const toImg = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left - ox) / sc, (e.clientY - r.top - oy) / sc]; };
let down = null;
cv.addEventListener('mousedown', e => { down = { x: e.clientX, y: e.clientY, ox, oy, moved: false }; });
window.addEventListener('mousemove', e => { if (!down) return; const dx = e.clientX - down.x, dy = e.clientY - down.y;
  if (Math.abs(dx) + Math.abs(dy) > 4) down.moved = true; if (down.moved) { ox = down.ox + dx; oy = down.oy + dy; draw(); } });
window.addEventListener('mouseup', e => { if (!down) return; const moved = down.moved; down = null; if (moved || e.target !== cv) return;
  const [x, y] = toImg(e), R = rec();
  if (e.shiftKey) { const all = [...R.players.map((p, i) => ['p', i, p[0], p[1]]), ...R.landmarks.map((m, i) => ['l', i, m.x, m.y])];
    let best = null, bd = 12 / sc; all.forEach(a => { const d = Math.hypot(a[2] - x, a[3] - y); if (d < bd) { bd = d; best = a; } });
    if (best) { (best[0] === 'p' ? R.players : R.landmarks).splice(best[1], 1); save(); } return; }
  if (mode === 'P') { R.players.push([+x.toFixed(1), +y.toFixed(1)]); save(); }
  else { pending = [x, y]; const r = cv.getBoundingClientRect(); form.style.left = Math.min(e.clientX - r.left + 10, r.width - 240) + 'px'; form.style.top = Math.min(e.clientY - r.top + 10, r.height - 170) + 'px'; form.style.display = 'block'; } });
cv.addEventListener('wheel', e => { e.preventDefault(); const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, f = Math.exp(-e.deltaY * 0.0015);
  ox = mx - (mx - ox) * f; oy = my - (my - oy) * f; sc *= f; draw(); }, { passive: false });
document.getElementById('lmok').onclick = () => { const R = rec(); R.landmarks.push({ x: +pending[0].toFixed(1), y: +pending[1].toFixed(1), line: document.getElementById('lmline').value, edge: document.getElementById('lmedge').value }); form.style.display = 'none'; save(); };
document.getElementById('lmcancel').onclick = () => { form.style.display = 'none'; };
const setMode = m => { mode = m; document.getElementById('mP').classList.toggle('on', m === 'P'); document.getElementById('mL').classList.toggle('on', m === 'L'); };
document.getElementById('mP').onclick = () => setMode('P'); document.getElementById('mL').onclick = () => setMode('L');
document.getElementById('prev').onclick = () => load(cur - 1); document.getElementById('next').onclick = () => load(cur + 1);
document.getElementById('fit').onclick = fit;
document.getElementById('undo').onclick = () => { const R = rec(); (mode === 'P' ? R.players : R.landmarks).pop(); save(); };
document.getElementById('dl').onclick = () => { const b = new Blob([JSON.stringify(S, null, 1)], { type: 'application/json' }), a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'click_labels.json'; a.click(); };
document.getElementById('imp').onchange = e => { const f = e.target.files[0]; if (!f) return; f.text().then(t => { S = JSON.parse(t); save(); }); };
window.addEventListener('keydown', e => { if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return; const k = e.key.toLowerCase();
  if (k === 'p') setMode('P'); else if (k === 'l') setMode('L'); else if (k === 'n') load(cur + 1); else if (k === 'b') load(cur - 1); else if (k === 'f') fit(); else if (k === 'z') document.getElementById('undo').click(); });
window.addEventListener('resize', resize); resize(); load(0);
</script></body></html>"""


def main():
    ids = json.loads((D / "manifest.json").read_text())
    (D / "index.html").write_text(HTML.replace("__IDS__", json.dumps(ids)).replace("__LINES__", json.dumps(LINES)))
    print("wrote", D / "index.html", len(ids), "frames")


if __name__ == "__main__":
    main()
