import base64, io, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle, Circle
from jinja2 import Environment
from weasyprint import HTML, default_url_fetcher
from shading import panel_footprints

MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_TH = ["à¸¡.à¸„.", "à¸.à¸ž.", "à¸¡à¸µ.à¸„.", "à¹€à¸¡.à¸¢.", "à¸ž.à¸„.", "à¸¡à¸´.à¸¢.", "à¸.à¸„.", "à¸ª.à¸„.", "à¸.à¸¢.", "à¸•.à¸„.", "à¸ž.à¸¢.", "à¸˜.à¸„."]

def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def chart_monthly(monthly):
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.bar(MONTHS_EN, [m["kwh"] for m in monthly], color="#f59e0b")
    ax.set_ylabel("kWh"); ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)

def chart_cashflow(rows):
    fig, ax = plt.subplots(figsize=(7, 2.4))
    c = [r["cumulative"] / 1000 for r in rows]
    ax.bar([r["year"] for r in rows], c, color=["#dc2626" if v < 0 else "#16a34a" for v in c])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("Year"); ax.set_ylabel("Cumulative cash flow (k THB)")
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)

def chart_layout(d, per_panel):
    b = d["building"]
    w, dep = float(b["w"]), float(b["d"])
    fig, ax = plt.subplots(figsize=(6, min(8, max(3, 6 * dep / w))))
    ax.add_patch(Rectangle((-w / 2, -dep / 2), w, dep, fc="#e5e7eb", ec="#374151", lw=1.5))
    vals = np.asarray(per_panel)
    norm = Normalize(min(vals.min(), vals.max() * 0.9), vals.max())
    cmap = plt.cm.RdYlGn
    for poly, v in zip(panel_footprints(d), vals):
        ax.add_patch(Polygon(poly, closed=True, fc=cmap(norm(v)), ec="#111", lw=.4))
    for o in d.get("obstacles", []):
        if o.get("type") == "tree":
            ax.add_patch(Circle((o["x"], -o["z"]), 2.5, fc="#4d7c0f", alpha=.35))
    ax.set_aspect("equal"); ax.margins(0.05); ax.autoscale_view()
    ax.set_xlabel("m (East â†’)"); ax.set_ylabel("m (North â†‘)")
    ax.annotate("N", xy=(0.95, 0.97), xytext=(0.95, 0.82), xycoords="axes fraction",
                ha="center", arrowprops=dict(arrowstyle="->", lw=1.5))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=.7, label="kWh / panel / year")
    return _png(fig)

def _safe_fetcher(url, *args, **kwargs):
    if url.startswith("data:"):
        return default_url_fetcher(url, *args, **kwargs)
    raise ValueError("external URL blocked")

env = Environment(autoescape=True)
env.filters["n"] = lambda v, d=0: f"{float(v):,.{d}f}"

TEMPLATE = env.from_string("""<!doctype html><html lang="th"><head><meta charset="utf-8"><style>
@page { size: A4; margin: 15mm 14mm 16mm;
  @bottom-right { content: "à¸«à¸™à¹‰à¸² " counter(page) " / " counter(pages); font-size: 8pt; color: #9ca3af } }
body { font-family: sans-serif; font-size: 10pt; color: #1f2937 }
h1 { font-size: 19pt; margin: 0 } h2 { font-size: 12.5pt; color: #b45309; border-bottom: 2px solid #f59e0b; padding-bottom: 2px; margin: 16px 0 6px }
.muted { color: #6b7280 } .note { font-size: 8.5pt; color: #6b7280 }
.kpis { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 }
.kpi { width: 31%; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 6px 8px; font-size: 9pt }
.kpi b { display: block; font-size: 14pt; color: #92400e }
table { width: 100%; border-collapse: collapse; font-size: 8.8pt; margin-top: 4px }
th, td { border: 1px solid #e5e7eb; padding: 3px 5px; text-align: right } th { background: #f3f4f6 }
td:first-child, th:first-child { text-align: left } img { max-width: 100% }
.pb { page-break-before: always }
</style></head><body>
{% if logo %}<img src="{{ logo }}" style="height:13mm;float:right">{% endif %}
<h1>Solar Report</h1>
<div class="muted">ID {{ pid[:8] }} â€¢ {{ today }}</div>
<div class="kpis">
  <div class="kpi">System Size<b>{{ s.kwp|n(2) }} kWp</b>{{ s.panels }} panels</div>
  <div class="kpi">Annual Energy<b>{{ s.annual_kwh|n }} kWh</b>{{ s.specific_yield|n }} kWh/kWp</div>
  <div class="kpi">First Year Saving<b>à¸¿{{ s.first_year_saving|n }}</b></div>
  <div class="kpi">Payback<b>{{ payback }}</b></div>
  <div class="kpi">Shading Loss<b>{{ (s.shading_loss*100)|n(1) }}%</b></div>
</div>
{% if snapshot %}<img src="{{ snapshot }}" style="max-height:78mm;display:block;margin:auto">{% endif %}
<h2>Monthly Energy</h2>
<img src="{{ img_monthly }}">
<h2 class="pb">Panel Layout</h2>
<img src="{{ img_layout }}" style="max-height:120mm;display:block;margin:auto">
<h2>Financial (25 Years)</h2>
<img src="{{ img_cash }}">
</body></html>""")

def render_pdf(d, r, pid):
    s = r["summary"]
    brand = d.get("brand") or {}
    snap = d.get("snapshot") if str(d.get("snapshot", "")).startswith("data:image/") else None
    logo = brand.get("logo") if str(brand.get("logo", "")).startswith("data:image/") else None
    html = TEMPLATE.render(
        p=d, s=s, r=r, a=r["assumptions"], pid=str(pid),
        payback=f'{s["payback"]:.1f} years' if s["payback"] else '25+ years',
        today=dt.date.today().strftime("%d/%m/%Y"), months=MONTHS_TH,
        brand_name=brand.get("name") or "SolarPlanner 3D", logo=logo,
        snapshot=snap, attribution=d.get("attribution"),
        img_monthly=chart_monthly(r["monthly"]),
        img_layout=chart_layout(d, r["per_panel_kwh"]),
        img_cash=chart_cashflow(r["finance"]["rows"]),
        fin_rows=[x for x in r["finance"]["rows"] if x["year"] in (1, 5, 10, 15, 20, 25)],
    )
    return HTML(string=html, url_fetcher=_safe_fetcher).write_pdf()
