$files = @{
    "render.yaml" = @"
services:
  - type: pserv
    name: solar-engine
    runtime: docker
    rootDir: engine
    region: singapore
    plan: standard
    envVars:
      - key: ENGINE_KEY
        generateValue: true
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_SERVICE_ROLE_KEY
        sync: false

  - type: web
    name: solar-web
    runtime: node
    rootDir: web
    region: singapore
    plan: starter
    buildCommand: npm ci && npm run build
    startCommand: npm start
    healthCheckPath: /api/health
    envVars:
      - key: NODE_VERSION
        value: "20"
      - key: ENGINE_HOSTPORT
        fromService: { type: pserv, name: solar-engine, property: hostport }
      - key: ENGINE_KEY
        fromService: { type: pserv, name: solar-engine, envVarKey: ENGINE_KEY }
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_SERVICE_ROLE_KEY
        sync: false
      - key: SUPABASE_ANON_KEY
        sync: false
      - key: GOOGLE_MAPS_KEY
        sync: false
      - key: STRIPE_SECRET_KEY
        sync: false
      - key: STRIPE_WEBHOOK_SECRET
        sync: false
"@

    "engine/requirements.txt" = @"
fastapi>=0.115
uvicorn[standard]>=0.30
pvlib>=0.11
pandas>=2.2
numpy>=1.26
requests>=2.32
supabase>=2.10
weasyprint>=63
jinja2>=3.1
matplotlib>=3.9
"@

    "engine/Dockerfile" = @"
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
      fonts-noto-core fonts-thai-tlwg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD exec uvicorn main:app --host 0.0.0.0 --port `${PORT:-8080}
"@

    "engine/pytest.ini" = @"
[pytest]
pythonpath = .
testpaths = tests
"@

    "engine/shading.py" = @'
import numpy as np

PANEL_W, PANEL_L, LIFT, NORMAL_OFFSET = 1.134, 2.278, 0.15, 0.02
SAMPLES_U = (-0.25, 0.25)
SAMPLES_W = (-1/3, 0.0, 1/3)

def sun_dirs(zenith_deg, azimuth_deg):
    el = np.radians(90 - np.asarray(zenith_deg, float))
    az = np.radians(np.asarray(azimuth_deg, float))
    return np.stack([np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)], axis=-1)

class Scene:
    def __init__(self, d):
        t, a = np.radians(float(d["tilt"])), np.radians(float(d["azimuth"]))
        b = d["building"]
        roof = float(b["h"])
        self.n = np.array([np.sin(t) * np.sin(a), np.cos(t), -np.sin(t) * np.cos(a)])
        self.u = np.array([-np.cos(a), 0.0, -np.sin(a)])
        self.w = np.array([np.cos(t) * np.sin(a), -np.sin(t), -np.cos(t) * np.cos(a)])
        yc = roof + LIFT + np.sin(t) * PANEL_L / 2
        self.centers = np.array([[float(p["x"]), yc, float(p["z"])] for p in d["panels"]])
        offs = np.array([su * PANEL_W * self.u + sw * PANEL_L * self.w
                         for su in SAMPLES_U for sw in SAMPLES_W])
        self.S = len(offs)
        self.samples = (self.centers[:, None, :] + offs[None] + self.n * NORMAL_OFFSET).reshape(-1, 3)
        self.owner = np.repeat(np.arange(len(self.centers)), self.S)
        self.boxes, self.spheres = [], []
        if b.get("solid", True):
            self.boxes.append((np.array([-b["w"] / 2, 0, -b["d"] / 2]),
                               np.array([b["w"] / 2, roof, b["d"] / 2])))
        for o in d.get("obstacles", []):
            if o.get("type") == "tree":
                x, z = float(o["x"]), float(o["z"])
                self.boxes.append((np.array([x - .3, 0, z - .3]), np.array([x + .3, 4, z + .3])))
                self.spheres.append((np.array([x, 5.5, z]), 2.5))
        hz = d.get("horizon")
        if hz and len(hz) == len(self.centers):
            self.horizon = np.clip(np.asarray(hz, float), 0, 90)
            self.diffuse_factor = np.mean(np.cos(np.radians(self.horizon)) ** 2, axis=1)
        else:
            self.horizon = None
            self.diffuse_factor = np.ones(len(self.centers))

def _blocked(sc, d):
    P, M = sc.samples, len(sc.samples)
    hit = np.zeros(M, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        for lo, hi in sc.boxes:
            t1, t2 = (lo - P) * inv, (hi - P) * inv
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit |= tmax >= np.maximum(tmin, 1e-6)
    for c, r in sc.spheres:
        oc = P - c
        b = oc @ d
        disc = b * b - (np.einsum("ij,ij->i", oc, oc) - r * r)
        hit |= (disc >= 0) & (-b + np.sqrt(np.maximum(disc, 0)) > 0)
    denom = d @ sc.n
    if denom > 1e-9 and len(sc.centers) > 1:
        t = ((sc.centers @ sc.n)[None, :] - (P @ sc.n)[:, None]) / denom
        q = P[:, None, :] + t[..., None] * d - sc.centers[None]
        inside = (np.abs(q @ sc.u) <= PANEL_W / 2) & (np.abs(q @ sc.w) <= PANEL_L / 2) & (t > 1e-6)
        inside[np.arange(M), sc.owner] = False
        hit |= inside.any(axis=1)
    return hit

def shade_fractions(sc, zen, az, res=1.0):
    zen, az = np.asarray(zen, float), np.asarray(az, float)
    key = np.stack([np.round(zen / res), np.round(az / res) % round(360 / res)], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    dirs = sun_dirs(uniq[:, 0] * res, uniq[:, 1] * res)
    N = len(sc.centers)
    frac = np.empty((len(uniq), N))
    for k, d in enumerate(dirs):
        frac[k] = _blocked(sc, d).reshape(N, sc.S).mean(axis=1)
    out = frac[inv.ravel()]
    if sc.horizon is not None:
        grid = np.arange(0, 365, 5)
        el = 90 - zen
        for i, prof in enumerate(sc.horizon):
            h = np.interp(az, grid, np.append(prof, prof[0]))
            out[:, i] = np.maximum(out[:, i], (el < h).astype(float))
    return out

def panel_footprints(d):
    sc = Scene(d)
    polys = []
    for c in sc.centers:
        pts = [c + su * PANEL_W / 2 * sc.u + sw * PANEL_L / 2 * sc.w
               for su, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        polys.append(np.array([[p[0], -p[2]] for p in pts]))
    return polys
'@

    "engine/simulate.py" = @'
import datetime as dt
from functools import lru_cache
import numpy as np
import pandas as pd
import pvlib
import requests
from shading import Scene, shade_fractions

NASA_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_PARAMS = {"ALLSKY_SFC_SW_DWN": "ghi", "ALLSKY_SFC_SW_DNI": "dni", "ALLSKY_SFC_SW_DIFF": "dhi",
               "T2M": "temp_air", "WS10M": "wind_speed"}

DEFAULTS = dict(
    tariff=4.5, export_rate=2.2, self_use=0.7, cost_per_kwp=30000, om_pct=0.01,
    degradation=0.005, escalation=0.02, years=25, inverter_kw=0.0,
    gamma=-0.0035, dc_losses=0.10, inv_eff=0.97, albedo=0.2, co2_kg_per_kwh=0.5,
    mount="open_rack_glass_polymer",
)
USER_KEYS = {"tariff": (0, 20), "export_rate": (0, 10), "self_use": (0, 1),
             "cost_per_kwp": (5000, 100000), "om_pct": (0, .05), "degradation": (0, .02),
             "escalation": (-.05, .1), "inverter_kw": (0, 10000)}

def _settings(d):
    f = dict(DEFAULTS)
    user = d.get("finance") or {}
    for k, (lo, hi) in USER_KEYS.items():
        if isinstance(user.get(k), (int, float)):
            f[k] = min(max(float(user[k]), lo), hi)
    if d.get("mount") in pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]:
        f["mount"] = d["mount"]
    return f

@lru_cache(maxsize=64)
def fetch_nasa_year(lat, lon, year):
    r = requests.get(NASA_URL, timeout=180, params={
        "parameters": ",".join(NASA_PARAMS), "community": "RE",
        "latitude": lat, "longitude": lon, "start": f"{year}0101", "end": f"{year}1231",
        "format": "JSON", "time-standard": "UTC"})
    r.raise_for_status()
    df = pd.DataFrame(r.json()["properties"]["parameter"]).rename(columns=NASA_PARAMS)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H", utc=True)
    df = df.replace(-999.0, np.nan)
    irr = ["ghi", "dni", "dhi"]
    df[irr] = df[irr].clip(lower=0).fillna(0)
    df[["temp_air", "wind_speed"]] = df[["temp_air", "wind_speed"]].interpolate(limit_direction="both")
    return df

def load_weather(lat, lon, n_years=3):
    lat, lon = round(lat, 3), round(lon, 3)
    last = dt.date.today().year - 1
    frames, used = [], []
    for y in range(last, last - n_years - 3, -1):
        if len(frames) == n_years:
            break
        try:
            df = fetch_nasa_year(lat, lon, y)
            if df.attrs.get("missing", 0) > 0.02 or len(df) < 8700:
                continue
            frames.append(df)
            used.append(y)
        except Exception as e:
            print("NASA error", y, e)
    if not frames:
        raise RuntimeError("NASA data fetch failed")
    return pd.concat(frames).sort_index(), sorted(used)

def finance(annual_kwh, kwp, f):
    capex = kwp * f["cost_per_kwp"]
    cum, payback, rows = -capex, None, []
    for y in range(1, int(f["years"]) + 1):
        e = annual_kwh * (1 - f["degradation"]) ** (y - 1)
        price = f["tariff"] * (1 + f["escalation"]) ** (y - 1)
        saving = e * f["self_use"] * price + e * (1 - f["self_use"]) * f["export_rate"] - capex * f["om_pct"]
        prev, cum = cum, cum + saving
        if payback is None and cum >= 0 and saving > 0:
            payback = (y - 1) + (-prev / saving)
        rows.append({"year": y, "energy_kwh": e, "saving": saving, "cumulative": cum})
    return {"capex": capex, "payback": payback, "first_year_saving": rows[0]["saving"],
            "total_saving": cum + capex, "rows": rows}

def run_simulation(d):
    f = _settings(d)
    lat, lon, tz = float(d["lat"]), float(d["lon"]), int(d.get("tz", 7))
    tilt, azim = float(d["tilt"]), float(d["azimuth"])
    n_pan = len(d["panels"])
    if n_pan == 0:
        raise ValueError("No panels")
    p_w = float(d["panelWatt"])
    kwp = n_pan * p_w / 1000

    wx, years = load_weather(lat, lon)
    n_years = len(years)
    times = wx.index

    sp = pvlib.solarposition.get_solarposition(times + pd.Timedelta(minutes=30), lat, lon)
    sp.index = times
    zen, saz = sp["apparent_zenith"], sp["azimuth"]

    poa = pvlib.irradiance.get_total_irradiance(
        tilt, azim, zen, saz, wx["dni"], wx["ghi"], wx["dhi"],
        dni_extra=pvlib.irradiance.get_extra_radiation(times),
        airmass=pvlib.atmosphere.get_relative_airmass(zen),
        albedo=f["albedo"], model="perez").fillna(0)
    iam = pvlib.iam.ashrae(pvlib.irradiance.aoi(tilt, azim, zen, saz)).fillna(0).values

    scene = Scene(d)
    shade = np.zeros((len(times), n_pan))
    up = (zen < 90).values & (poa["poa_direct"].values > 0)
    if up.any():
        shade[up] = shade_fractions(scene, zen.values[up], saz.values[up])

    beam = (poa["poa_direct"].values * iam)[:, None]
    diff = poa["poa_diffuse"].values[:, None]
    eff = beam * (1 - shade) + diff * scene.diffuse_factor[None, :]
    eff_ref = beam + diff

    tp = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][f["mount"]]
    tcell = pvlib.temperature.sapm_cell(poa["poa_global"], wx["temp_air"], wx["wind_speed"], **tp).values[:, None]

    dc = np.clip(pvlib.pvsystem.pvwatts_dc(eff, tcell, p_w, f["gamma"]), 0, None) * (1 - f["dc_losses"])
    dc_ref = np.clip(pvlib.pvsystem.pvwatts_dc(eff_ref, tcell, p_w, f["gamma"]), 0, None) * (1 - f["dc_losses"])
    inv_w = f["inverter_kw"] * 1000 if f["inverter_kw"] > 0 else np.inf
    ac = np.minimum(dc.sum(axis=1) * f["inv_eff"], inv_w)
    ac_ref = np.minimum(dc_ref[:, 0] * n_pan * f["inv_eff"], inv_w)

    local = times.tz_convert(f"Etc/GMT{-tz:+d}")
    ac_kwh = pd.Series(ac / 1000, index=local)
    monthly_kwh = (ac_kwh.groupby(local.month).sum() / n_years).reindex(range(1, 13), fill_value=0)
    annual = float(monthly_kwh.sum())
    annual_ref = float(ac_ref.sum() / 1000 / n_years)
    daily_ghi = pd.Series(wx["ghi"].values / 1000, index=local).resample("D").sum()
    ghi_m = daily_ghi.groupby(daily_ghi.index.month).mean()
    poa_annual = float(poa["poa_global"].sum() / 1000 / n_years)
    per_panel = dc.sum(axis=0) * f["inv_eff"] / 1000 / n_years

    fin = finance(annual, kwp, f)
    return {
        "summary": {
            "kwp": kwp, "panels": n_pan, "annual_kwh": annual,
            "specific_yield": annual / kwp,
            "shading_loss": 1 - annual / annual_ref if annual_ref > 0 else 0.0,
            "pr": annual / (kwp * poa_annual) if poa_annual > 0 else 0.0,
            "poa_kwh_m2": poa_annual,
            "first_year_saving": fin["first_year_saving"], "payback": fin["payback"],
            "capex": fin["capex"], "total_saving": fin["total_saving"],
            "co2_t": annual * f["co2_kg_per_kwh"] / 1000, "years": years,
        },
        "monthly": [{"month": m, "ghi": float(ghi_m.get(m, 0)), "kwh": float(monthly_kwh[m])}
                    for m in range(1, 13)],
        "per_panel_kwh": [float(v) for v in per_panel],
        "finance": fin,
        "assumptions": f,
    }
'@

    "engine/report.py" = @'
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
MONTHS_TH = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]

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
    ax.set_xlabel("m (East →)"); ax.set_ylabel("m (North ↑)")
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
  @bottom-right { content: "หน้า " counter(page) " / " counter(pages); font-size: 8pt; color: #9ca3af } }
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
<div class="muted">ID {{ pid[:8] }} • {{ today }}</div>
<div class="kpis">
  <div class="kpi">System Size<b>{{ s.kwp|n(2) }} kWp</b>{{ s.panels }} panels</div>
  <div class="kpi">Annual Energy<b>{{ s.annual_kwh|n }} kWh</b>{{ s.specific_yield|n }} kWh/kWp</div>
  <div class="kpi">First Year Saving<b>฿{{ s.first_year_saving|n }}</b></div>
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
'@

    "engine/main.py" = @'
import os, hmac, traceback
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, Body
from fastapi.responses import Response
from pydantic import BaseModel
from supabase import create_client
from simulate import run_simulation
from report import render_pdf

ENGINE_KEY = os.environ["ENGINE_KEY"]
sb = (create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
      if os.getenv("SUPABASE_URL") else None)
app = FastAPI(title="Solar Engine")

def check_key(key: str):
    if not hmac.compare_digest(key or "", ENGINE_KEY):
        raise HTTPException(401, "bad key")

class Job(BaseModel):
    project_id: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/report", status_code=202)
def report(job: Job, bg: BackgroundTasks, x_engine_key: str = Header("")):
    check_key(x_engine_key)
    claimed = (sb.table("projects").update({"status": "processing", "error": None})
               .eq("id", job.project_id).eq("status", "paid").execute())
    if not claimed.data:
        return {"skipped": True}
    bg.add_task(process, job.project_id)
    return {"queued": True}

def process(pid: str):
    try:
        p = sb.table("projects").select("user_id,data").eq("id", pid).single().execute().data
        result = run_simulation(p["data"])
        pdf = render_pdf(p["data"], result, pid)
        path = f"{p['user_id']}/{pid}.pdf"
        sb.storage.from_("reports").upload(path, pdf, {"content-type": "application/pdf", "upsert": "true"})
        sb.table("projects").update({
            "status": "done", "pdf_path": path,
            "result": {"summary": result["summary"], "monthly": result["monthly"]},
        }).eq("id", pid).execute()
    except Exception as e:
        traceback.print_exc()
        sb.table("projects").update({"status": "error", "error": str(e)[:500]}).eq("id", pid).execute()

@app.post("/preview")
def preview(project: dict = Body(...), x_engine_key: str = Header("")):
    check_key(x_engine_key)
    result = run_simulation(project)
    return Response(render_pdf(project, result, "PREVIEW"), media_type="application/pdf")
'@

    "engine/tests/test_engine.py" = @'
import os
KEY = os.environ.setdefault("ENGINE_KEY", "test")
import numpy as np, pandas as pd, pvlib, pytest
from fastapi.testclient import TestClient
import simulate, main
from shading import Scene, shade_fractions, sun_dirs
from simulate import run_simulation, finance, DEFAULTS
from report import render_pdf

BASE = {"lat": 13.7563, "lon": 100.5018, "tz": 7,
        "building": {"w": 12, "d": 10, "h": 6},
        "tilt": 15, "azimuth": 180, "panelWatt": 550,
        "panels": [{"x": x, "z": 0} for x in (-3, -1.8, -0.6, 0.6)],
        "obstacles": []}

def fake_weather(lat, lon, n_years=3):
    t = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
    cs = pvlib.location.Location(lat, lon).get_clearsky(t + pd.Timedelta(minutes=30), model="ineichen", linke_turbidity=4.0)
    cs.index = t
    wx = cs[["ghi", "dni", "dhi"]] * 0.75
    wx["temp_air"] = 30.0
    wx["wind_speed"] = 2.0
    return wx, [2023]

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(simulate, "load_weather", fake_weather)

def test_sun_direction_axes():
    np.testing.assert_allclose(sun_dirs(0, 0), [0, 1, 0], atol=1e-9)

def test_simulation_plausible():
    r = run_simulation(BASE)
    assert r["summary"]["specific_yield"] > 1000

def test_api():
    c = TestClient(main.app)
    assert c.get("/health").json() == {"ok": True}
'@

    "supabase/schema.sql" = @"
create table public.projects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users on delete set null,
  data jsonb not null,
  status text not null default 'draft' check (status in ('draft','paid','processing','done','error')),
  paid_via text,
  result jsonb,
  pdf_path text,
  error text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table public.credit_balances (
  user_id uuid primary key references auth.users on delete cascade,
  balance int not null default 0 check (balance >= 0)
);

create table public.credit_ledger (
  id bigserial primary key,
  user_id uuid not null references auth.users on delete cascade,
  delta int not null,
  reason text not null,
  stripe_session_id text unique,
  project_id uuid references public.projects,
  created_at timestamptz default now()
);

alter table projects enable row level security;
alter table credit_balances enable row level security;
alter table credit_ledger enable row level security;

create or replace function public.add_credits(p_user uuid, p_amount int, p_session text)
returns void language plpgsql security definer set search_path = public as $$
begin
  insert into credit_ledger(user_id, delta, reason, stripe_session_id) values (p_user, p_amount, 'purchase', p_session);
  insert into credit_balances(user_id, balance) values (p_user, p_amount)
  on conflict (user_id) do update set balance = credit_balances.balance + excluded.balance;
exception when unique_violation then null;
end $$;

create or replace function public.use_credit(p_user uuid, p_project uuid)
returns boolean language plpgsql security definer set search_path = public as $$
begin
  update credit_balances set balance = balance - 1 where user_id = p_user and balance > 0;
  if not found then return false; end if;
  update projects set status = 'paid', paid_via = 'credit', updated_at = now() where id = p_project and user_id = p_user and status = 'draft';
  if not found then raise exception 'not found'; end if;
  insert into credit_ledger(user_id, delta, reason, project_id) values (p_user, -1, 'report', p_project);
  return true;
end $$;
"@

    "web/next.config.ts" = @"
import type { NextConfig } from 'next';
const nextConfig: NextConfig = {
  async rewrites() { return [{ source: '/', destination: '/index.html' }]; },
};
export default nextConfig;
"@

    "web/lib/server.ts" = @"
import Stripe from 'stripe';
import { createClient } from '@supabase/supabase-js';

export const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!);
export const db = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!, { auth: { persistSession: false } });
export const SITE = process.env.SITE_URL ?? process.env.RENDER_EXTERNAL_URL!;
const ENGINE_URL = process.env.ENGINE_URL ?? `http://${process.env.ENGINE_HOSTPORT}`;

export const PRODUCTS = {
  single: { name: 'Report 1', amount: 9900, credits: 0 },
  pack10: { name: '10 Credits', amount: 79000, credits: 10 },
  pack30: { name: '30 Credits', amount: 199000, credits: 30 },
} as const;
export type ProductId = keyof typeof PRODUCTS;
export const isProduct = (p: unknown): p is ProductId => typeof p === 'string' && Object.hasOwn(PRODUCTS, p);

export const json = (body: unknown, status = 200) => Response.json(body, { status });

export async function getUser(req: Request) {
  const token = req.headers.get('authorization')?.replace(/^Bearer /, '');
  if (!token) return null;
  const { data } = await db.auth.getUser(token);
  return data.user ?? null;
}

export function triggerEngine(projectId: string) {
  return fetch(`${ENGINE_URL}/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Engine-Key': process.env.ENGINE_KEY! },
    body: JSON.stringify({ project_id: projectId }),
  }).catch(e => console.error('engine trigger failed', e));
}

export function validateProject(p: any): string | null {
  const num = (v: any) => typeof v === 'number' && Number.isFinite(v);
  if (!num(p?.lat) || Math.abs(p.lat) > 90 || !num(p?.lon) || Math.abs(p.lon) > 180) return 'Invalid lat/lon';
  if (!Array.isArray(p.panels) || p.panels.length < 1 || p.panels.length > 1000) return 'Invalid panels';
  return null;
}
"@

    "web/app/config.js/route.ts" = @"
export const dynamic = 'force-dynamic';
export function GET() {
  const cfg = {
    SUPABASE_URL: process.env.SUPABASE_URL ?? '',
    SUPABASE_ANON_KEY: process.env.SUPABASE_ANON_KEY ?? '',
    GOOGLE_KEY: process.env.GOOGLE_MAPS_KEY ?? '',
    API_BASE: '',
  };
  return new Response(`export default ${JSON.stringify(cfg)};`, {
    headers: { 'Content-Type': 'text/javascript; charset=utf-8', 'Cache-Control': 'no-store' },
  });
}
"@

    "web/app/api/health/route.ts" = 'export const GET = () => Response.json({ ok: true });'

    "web/app/api/projects/route.ts" = @"
import { db, getUser, json, validateProject } from '@/lib/server';
export async function POST(req: Request) {
  const user = await getUser(req);
  if (!user) return json({ error: 'Unauthorized' }, 401);
  const data = await req.json();
  const err = validateProject(data);
  if (err) return json({ error: err }, 400);
  const { data: p, error } = await db.from('projects').insert({ user_id: user.id, data }).select('id').single();
  if (error) return json({ error: error.message }, 500);
  return json({ id: p.id });
}
"@

    "web/app/api/checkout/route.ts" = @"
import { db, getUser, isProduct, json, PRODUCTS, SITE, stripe } from '@/lib/server';
export async function POST(req: Request) {
  const user = await getUser(req);
  if (!user) return json({ error: 'Unauthorized' }, 401);
  const { product, projectId } = await req.json();
  if (!isProduct(product)) return json({ error: 'Invalid product' }, 400);
  const item = PRODUCTS[product];
  const session = await stripe.checkout.sessions.create({
    mode: 'payment',
    payment_method_types: ['card', 'promptpay'],
    customer_email: user.email,
    line_items: [{ quantity: 1, price_data: { currency: 'thb', unit_amount: item.amount, product_data: { name: item.name } } }],
    metadata: { user_id: user.id, product, project_id: projectId ?? '' },
    success_url: product === 'single' ? `${SITE}/report.html?id=${projectId}` : `${SITE}/?credits=ok`,
    cancel_url: `${SITE}/`,
  });
  return json({ url: session.url });
}
"@

    "web/app/api/webhook/route.ts" = @"
import type Stripe from 'stripe';
import { db, isProduct, PRODUCTS, stripe, triggerEngine } from '@/lib/server';
export async function POST(req: Request) {
  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(await req.text(), req.headers.get('stripe-signature')!, process.env.STRIPE_WEBHOOK_SECRET!);
  } catch { return new Response('bad signature', { status: 400 }); }
  if (event.type === 'checkout.session.completed' || event.type === 'checkout.session.async_payment_succeeded') {
    const s = event.data.object as Stripe.Checkout.Session;
    if (s.payment_status !== 'paid') return new Response('pending');
    const { user_id, product, project_id } = s.metadata ?? {};
    if (!isProduct(product)) return new Response('unknown product');
    if (product === 'single') {
      const { data } = await db.from('projects').update({ status: 'paid', paid_via: 'stripe', updated_at: new Date().toISOString() }).eq('id', project_id).eq('status', 'draft').select('id');
      if (data?.length) await triggerEngine(project_id!);
    } else {
      await db.rpc('add_credits', { p_user: user_id, p_amount: PRODUCTS[product].credits, p_session: s.id });
    }
  }
  return new Response('ok');
}
"@

    "web/app/api/redeem/route.ts" = @"
import { db, getUser, json, triggerEngine } from '@/lib/server';
export async function POST(req: Request) {
  const user = await getUser(req);
  if (!user) return json({ error: 'Unauthorized' }, 401);
  const { projectId } = await req.json();
  const { data: ok, error } = await db.rpc('use_credit', { p_user: user.id, p_project: projectId });
  if (error || !ok) return json({ error: 'Insufficient credits' }, 400);
  await triggerEngine(projectId);
  return json({ ok: true });
}
"@

    "web/app/api/me/route.ts" = @"
import { db, getUser, json } from '@/lib/server';
export async function GET(req: Request) {
  const user = await getUser(req);
  if (!user) return json({ error: 'Unauthorized' }, 401);
  const { data: bal } = await db.from('credit_balances').select('balance').eq('user_id', user.id).maybeSingle();
  const { data: projects } = await db.from('projects').select('id,status,created_at').eq('user_id', user.id).neq('status', 'draft').order('created_at', { ascending: false }).limit(20);
  return json({ email: user.email, balance: bal?.balance ?? 0, projects });
}
"@

    "web/app/api/report/[id]/route.ts" = @"
import { db, getUser, json, triggerEngine } from '@/lib/server';
export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const user = await getUser(req);
  if (!user) return json({ error: 'Unauthorized' }, 401);
  const { data: p } = await db.from('projects').select('status,pdf_path,result,error').eq('id', id).eq('user_id', user.id).maybeSingle();
  if (!p) return json({ error: 'Not found' }, 404);
  if (p.status === 'paid') await triggerEngine(id);
  let url: string | null = null;
  if (p.status === 'done' && p.pdf_path) {
    const { data } = await db.storage.from('reports').createSignedUrl(p.pdf_path, 600);
    url = data?.signedUrl ?? null;
  }
  return json({ status: p.status, url, summary: p.result?.summary ?? null });
}
"@

    "web/public/report.html" = @"
<!DOCTYPE html>
<html lang="th"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Report</title>
<style>body{font-family:system-ui,sans-serif;max-width:560px;margin:60px auto;padding:0 16px;text-align:center;color:#1f2937}a.btn{display:inline-block;margin-top:16px;padding:12px 24px;background:#16a34a;color:#fff;border-radius:8px;text-decoration:none;font-weight:600}</style></head>
<body><h2 id="msg">Checking...</h2><p id="sub"></p><a id="dl" class="btn" style="display:none" target="_blank">Download PDF</a><p><a href="/">← Back</a></p>
<script type="module">
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import CONFIG from './config.js';
const supa = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_ANON_KEY);
const id = new URLSearchParams(location.search).get('id');
const $ = i => document.getElementById(i);
async function poll() {
  const { data: { session } } = await supa.auth.getSession();
  if (!session) { $('msg').textContent = 'Please login first'; return; }
  const r = await fetch(`/api/report/${id}`, { headers: { Authorization: `Bearer ${session.access_token}` } });
  const j = await r.json();
  if (!r.ok) { $('msg').textContent = j.error || 'Not found'; return; }
  if (j.status === 'done') {
    $('msg').textContent = 'Report Ready! 🎉';
    $('sub').textContent = `Annual Energy: ${Math.round(j.summary.annual_kwh).toLocaleString()} kWh`;
    $('dl').href = j.url; $('dl').style.display = 'inline-block';
    return;
  }
  $('msg').textContent = 'Processing... (' + j.status + ')';
  if (j.status !== 'error') setTimeout(poll, 4000);
}
poll();
</script></body></html>
"@

    "web/public/manifest.webmanifest" = @"
{
  "name": "SolarPlanner 3D",
  "short_name": "SolarPlanner",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0f172a",
  "theme_color": "#f59e0b",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
"@

    ".github/workflows/ci.yml" = @"
name: CI
on: [push, pull_request]
jobs:
  engine:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t solar-engine ./engine
      - run: docker run --rm solar-engine sh -c "pip install -q pytest httpx && pytest -q"
"@
}

foreach ($path in $files.Keys) {
    $dir = Split-Path -Parent $path
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
    Set-Content -Path $path -Value $files[$path] -Encoding utf8
    Write-Host "Created: $path"
}
Write-Host "`nAll files created successfully!" -ForegroundColor Green