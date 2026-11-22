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
