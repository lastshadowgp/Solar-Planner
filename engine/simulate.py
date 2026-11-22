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
