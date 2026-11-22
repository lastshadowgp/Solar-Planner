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
