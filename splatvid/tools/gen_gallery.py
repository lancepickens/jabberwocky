import base64, cv2, html, os

S = "/private/tmp/claude-501/-Users-rho-dev-jabberwocky-splatvid/48ec706e-939d-4ef1-ab8c-6f179ed29900/scratchpad"


def img(path, maxw):
    im = cv2.imread(os.path.join(S, path))
    h, w = im.shape[:2]
    if w > maxw:
        im = cv2.resize(im, (maxw, round(h * maxw / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def fig_single(src, cap, maxw=1120):
    return (f'<figure class="single"><div class="frame"><img loading="lazy" src="{img(src, maxw)}" alt="{html.escape(cap)}"></div>'
            f'<figcaption>{cap}</figcaption></figure>')


def fig_pair(a, la, b, lb, maxw=760):
    def cell(src, lab, tone):
        return (f'<figure class="cell"><span class="tag {tone}">{lab}</span>'
                f'<div class="frame"><img loading="lazy" src="{img(src, maxw)}" alt="{html.escape(lab)}"></div></figure>')
    return f'<div class="pair">{cell(a, la, "before")}{cell(b, lb, "after")}</div>'


SECTIONS = [
    ("00", "Input", "One handheld phone clip",
     "A 28-second iPhone video orbiting a YETI Hopper Flip cooler on a cluttered table. Only 39 of 100 sampled frames were usable. Everything below is reconstructed from these pixels alone — no depth sensor, no turntable, no markers.",
     fig_pair("reg_40.jpg", "Source frame — the metric anchor", "depth_f40.jpg", "DepthAnything v2 — near is bright", 620)),

    ("01", "Live viewer · framing", "From lost to framed",
     "The in-browser WebGL viewer auto-framed on a mean center and max-distance radius, so a few far floaters shrank the object to an off-center speck. A median center plus outlier culling fixed it — a real bug in what users actually see.",
     fig_pair("viewer_shot.png", "Before — object lost to a dot", "viewer_shot4.png", "After — centered and framed")),

    ("02", "Live viewer · antialiasing", "Smoother splats, less popping",
     "Real-time shader work, all in the browser: render splats to 2.5σ instead of a hard 2σ cutoff, feather the silhouette by one screen-pixel with fwidth(), and tighten the re-sort threshold to cut depth-order popping while orbiting.",
     fig_single("viewer_before_after.jpg", "Same scene, same view — hard 2σ edges (left) vs 2.5σ + antialiasing (right)")),

    ("03", "Neural renderer", "A U-Net shader over the splat",
     "A deferred neural renderer: splat learned features, decode with a small U-Net trained on perceptual + temporal losses. Same 24k-gaussian geometry, same viewpoint — the shader smooths the raw rasterizer's jagged splat edges into coherent surfaces.",
     fig_single("neural_compare.jpg", "Raw gaussians (left) vs the neural shader on identical geometry (right)")),

    ("04", "Photogrammetry mesh", "From splat to surface",
     "TSDF-fusing the trained splat's depth from every recovered camera turns the point-like splat into a real triangle mesh — the cooler becomes a solid object, the table a continuous surface, colored from the reconstruction.",
     fig_single("mesh_render.jpg", "Textured photogrammetry mesh of the scene, rendered from a captured viewpoint", 640)),

    ("05", "Mesh quality · monocular depth", "Cleaner surfaces from independent depth",
     "Fusing the mesh from DepthAnything's dense monocular depth (with the real photo as color) instead of the noisy splat depth. Because the monocular depth is smooth and independent of the splat's floaters, the surface is markedly better.",
     fig_single("mesh_quality_cmp.jpg", "Mesh from splat depth (left) vs from monocular depth (right) — smoother, more complete, fewer holes")),

    ("06", "Web transport · Draco", "17× smaller, visually lossless",
     "For internet delivery the mesh is the bottleneck. gzip barely helps (1.3× — binary floats don't compress). Draco at 11-bit position quantization takes the 7.76 MB mesh down to 0.46 MB with no visible change; it decodes in-browser via three.js.",
     fig_pair("mesh_mono_f40.jpg", "Original mesh · 7.76 MB", "mesh_draco11_f40.jpg", "Draco 11-bit · 0.46 MB (17×)", 620)),
]

FINDINGS = [
    ("Metric scale from a known object", "YETI cooler (30 cm) → whole scene 2.5 m, orbit 0.76 m", "win"),
    ("Fragmented-video SfM (bridging + largest component)", "hard crash → 39 cameras reconstructed", "win"),
    ("Monocular depth for mesh fusion", "markedly cleaner surface", "win"),
    ("Draco mesh compression", "17× smaller, visually lossless", "win"),
    ("Batched pure-PyTorch rasterizer", "numerically identical but slower — reverted", "null"),
    ("Depth supervision → floaters", "no reduction across 3 experiments (p99 +4.3%)", "neg"),
    ("Depth supervision → novel-view quality", "worse: −3.7 dB PSNR, +0.22 LPIPS held-out", "neg"),
]

body = []
for num, eyebrow, title, desc, figs in SECTIONS:
    body.append(f'''<section class="phase">
  <div class="lede">
    <p class="eyebrow"><span class="num">{num}</span> {eyebrow}</p>
    <h2>{title}</h2>
    <p class="desc">{desc}</p>
  </div>
  {figs}
</section>''')

rows = ""
label = {"win": "kept", "null": "reverted", "neg": "dropped"}
for name, res, tone in FINDINGS:
    rows += (f'<tr><td>{html.escape(name)}</td><td class="res">{html.escape(res)}</td>'
             f'<td><span class="v v-{tone}">{label[tone]}</span></td></tr>')

findings = f'''<section class="phase findings">
  <div class="lede">
    <p class="eyebrow"><span class="num">✳</span> The honest ledger</p>
    <h2>What worked, what didn't</h2>
    <p class="desc">Every experiment logged — including the negative results, which are the point: they tell you what not to chase. Depth was great for the mesh and a net loss for the splat.</p>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>Experiment</th><th>Result</th><th>Verdict</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</section>'''

STYLE = '''<style>
:root{
  --bg:#0b0e14; --panel:#121824; --panel2:#0e131c; --line:#232c3b;
  --ink:#e8edf5; --muted:#8b97a8; --lime:#c9f542; --lime-dim:#9dc233;
  --warn:#e6a94a; --neg:#e77b6b;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.6;-webkit-font-smoothing:antialiased;
  background-image:radial-gradient(1200px 600px at 70% -10%,rgba(201,245,66,.05),transparent 60%);}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
header.hero{padding:88px 24px 56px;max-width:1120px;margin:0 auto}
.kicker{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--lime-dim);margin:0 0 20px}
h1{font-size:clamp(38px,6vw,68px);line-height:1.02;letter-spacing:-.025em;font-weight:800;
  margin:0;text-wrap:balance;max-width:16ch}
h1 em{font-style:normal;color:var(--lime)}
.thesis{font-size:clamp(17px,2.1vw,21px);color:var(--muted);max-width:60ch;margin:22px 0 0}
.meta{display:flex;flex-wrap:wrap;gap:8px 10px;margin-top:30px;font-family:var(--mono);font-size:12px}
.chip{border:1px solid var(--line);border-radius:999px;padding:5px 11px;color:var(--muted);
  background:var(--panel2)}
.chip b{color:var(--ink);font-weight:600}
.rule{height:1px;background:linear-gradient(90deg,var(--lime),transparent);max-width:1072px;
  margin:0 auto;opacity:.5}
.phase{padding:64px 0;border-bottom:1px solid var(--line)}
.phase:last-of-type{border-bottom:0}
.lede{max-width:66ch;margin-bottom:30px}
.eyebrow{font-family:var(--mono);font-size:12.5px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted);margin:0 0 12px;display:flex;align-items:center;gap:12px}
.num{color:var(--lime);font-weight:600}
h2{font-size:clamp(24px,3.4vw,34px);letter-spacing:-.02em;font-weight:750;margin:0 0 12px;text-wrap:balance}
.desc{color:var(--muted);margin:0;font-size:16.5px}
.frame{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#05070b;
  box-shadow:0 24px 60px -30px rgba(0,0,0,.8)}
.frame img{display:block;width:100%;height:auto}
figure{margin:0}
.single figcaption,figure.cell{margin-top:0}
.single figcaption{margin-top:12px;font-family:var(--mono);font-size:12.5px;color:var(--muted);
  letter-spacing:.01em}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.cell{display:flex;flex-direction:column;gap:10px}
.tag{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);display:inline-flex;align-items:center;gap:8px}
.tag::before{content:"";width:8px;height:8px;border-radius:2px;background:var(--muted)}
.tag.before::before{background:var(--warn)}
.tag.after{color:var(--lime-dim)}
.tag.after::before{background:var(--lime)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel2)}
table{border-collapse:collapse;width:100%;font-size:14.5px;min-width:600px}
th,td{text-align:left;padding:13px 18px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:var(--mono);font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);font-weight:600}
tbody tr:last-child td{border-bottom:0}
td.res{color:var(--muted)}
.v{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  padding:3px 9px;border-radius:999px;border:1px solid;white-space:nowrap}
.v-win{color:var(--lime);border-color:rgba(201,245,66,.35);background:rgba(201,245,66,.08)}
.v-null{color:var(--warn);border-color:rgba(230,169,74,.35);background:rgba(230,169,74,.07)}
.v-neg{color:var(--neg);border-color:rgba(231,123,107,.35);background:rgba(231,123,107,.08)}
footer{padding:56px 24px 80px;max-width:1120px;margin:0 auto;color:var(--muted);
  font-family:var(--mono);font-size:12.5px;letter-spacing:.02em}
footer b{color:var(--ink)}
@media(max-width:640px){.pair{grid-template-columns:1fr}header.hero{padding:56px 24px 40px}}
</style>'''

HTML = STYLE + f'''
<header class="hero">
  <p class="kicker">Computational photography · results log</p>
  <h1>One phone video, taken <em>apart</em> and rebuilt.</h1>
  <p class="thesis">A 28-second clip of a cooler on a cluttered table, turned into gaussian splats, a metric-scaled mesh, and a web-ready model — every step measured, every dead end recorded.</p>
  <div class="meta">
    <span class="chip"><b>39</b> cameras recovered</span>
    <span class="chip"><b>97k</b> gaussians</span>
    <span class="chip"><b>2.5 m</b> scene, metric</span>
    <span class="chip"><b>17×</b> mesh compression</span>
    <span class="chip"><b>5</b> pull requests</span>
  </div>
</header>
<div class="rule"></div>
<main class="wrap">
{''.join(body)}
{findings}
</main>
<footer>Reconstructed from a single handheld iPhone clip · <b>splatvid</b> — SfM, differentiable rasterizer, neural shader, TSDF mesh, monocular depth &amp; Draco, all from scratch.</footer>
'''

open(os.path.join(S, "gallery.html"), "w").write(HTML)
print("wrote gallery.html", round(len(HTML) / 1e6, 2), "MB")
