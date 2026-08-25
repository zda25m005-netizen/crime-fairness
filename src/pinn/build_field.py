"""
STEP 1 for the PINN — turn point crime records into a continuous field u(x, y, t)
================================================================================
A PINN solves a PDE, and a PDE acts on a FIELD: a quantity defined over
continuous space and time.  The dataset used so far is a table
(neighbourhood x category x day), which a PDE cannot touch.  This script
rebuilds the raw data into the shape a PDE needs.

What it produces
----------------
    U        (T, H, W)  crime intensity per grid cell per time step
    x, y     (H, W)     cell centre coordinates, normalised to [-1, 1]
    t        (T,)       time, normalised to [0, 1]
    mask     (H, W)     True where the cell is inside the city
    area     (H, W)     Chicago community area id per cell  (77 areas)
    group    (H, W)     'Head' / 'Mid' / 'Tail' per cell, by total crime

Why the last two matter: they carry the fairness analysis over to the PINN.
Without them you can measure accuracy but not who the model serves well.

Design choices worth knowing
----------------------------
* Coordinates are normalised. PINNs train badly on raw latitude/longitude
  because the derivative scales are tiny; [-1, 1] is standard practice.
* Optional Gaussian smoothing. Raw daily counts per cell are extremely spiky,
  and a PDE describes a smooth field. Smoothing is a modelling assumption and
  is recorded in the output so it can be reported honestly.
* Cells that never see a crime all year are outside the city boundary and are
  masked out, not treated as true zeros.
* The train/val/test split is by TIME, never random, matching the rest of the
  project. Random splits would leak the future into training.

Usage
-----
    python build_field.py --year 2015 --grid 48 --freq D --smooth 1.0
    python build_field.py --year 2015 --grid 64 --freq W --smooth 0.8
"""
from __future__ import annotations

import argparse
import json
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd

RESOURCE = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
# Chicago bounding box, trimmed to the mainland city (a few records land in
# the lake or are mis-geocoded to 0,0 -- those are dropped).
LAT_LO, LAT_HI = 41.644, 42.023
LON_LO, LON_HI = -87.940, -87.524


# --------------------------------------------------------------------------- #
# 1. fetch
# --------------------------------------------------------------------------- #
def fetch(year0: int, year1: int, category: str | None = None,
          page: int = 50_000, max_rows: int = 2_000_000) -> pd.DataFrame:
    """Page the Socrata endpoint for a RANGE of years.

    The category filter is pushed server-side. Without it, five years of all
    crime types is ~1.3M rows; with it, burglary alone is ~65k. That is the
    difference between a 30-second fetch and a 20-minute one.
    """
    where = (f"date >= '{year0}-01-01T00:00:00.000' "
             f"AND date <= '{year1}-12-31T23:59:59.000' "
             f"AND latitude IS NOT NULL")
    if category:
        where += f" AND upper(primary_type) = '{category.upper()}'"
    out, offset = [], 0
    while offset < max_rows:
        q = urlencode({"$select": "date,latitude,longitude,community_area,primary_type",
                       "$where": where, "$limit": page, "$offset": offset})
        chunk = pd.read_json(f"{RESOURCE}?{q}")
        if chunk.empty:
            break
        out.append(chunk)
        offset += page
        print(f"  fetched {offset} rows...", flush=True)
        time.sleep(0.4)
    if not out:
        raise SystemExit("No rows returned - check the year or the portal.")
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2. smoothing without scipy (separable Gaussian, reflect padding)
# --------------------------------------------------------------------------- #
def gaussian_blur(A: np.ndarray, sigma: float) -> np.ndarray:
    """Blur the last two axes of A. sigma is in grid cells; 0 disables."""
    if sigma <= 0:
        return A
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = A.astype(np.float64)
    for axis in (-2, -1):
        out = np.apply_along_axis(
            lambda v: np.convolve(np.pad(v, r, mode="reflect"), k, mode="valid"),
            axis, out)
    return out


# --------------------------------------------------------------------------- #
# 3. build
# --------------------------------------------------------------------------- #
def build(args):
    df = (pd.read_csv(args.csv) if args.csv
          else fetch(args.year0, args.year1, args.category))
    n0 = len(df)
    df = df.dropna(subset=["date", "latitude", "longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df.latitude.between(LAT_LO, LAT_HI)) &
            (df.longitude.between(LON_LO, LON_HI))]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if args.category:                       # optional single-category field
        df = df[df["primary_type"].str.upper() == args.category.upper()]
    print(f"\nkept {len(df)} of {n0} records "
          f"({100*len(df)/max(n0,1):.1f}%) after cleaning")

    # ---- spatial grid ----------------------------------------------------- #
    G = args.grid
    lat_edges = np.linspace(LAT_LO, LAT_HI, G + 1)
    lon_edges = np.linspace(LON_LO, LON_HI, G + 1)
    iy = np.clip(np.digitize(df.latitude.values, lat_edges) - 1, 0, G - 1)
    ix = np.clip(np.digitize(df.longitude.values, lon_edges) - 1, 0, G - 1)

    # ---- temporal bins ---------------------------------------------------- #
    per = df["date"].dt.to_period({"D": "D", "W": "W", "M": "M"}[args.freq])
    periods = np.array(sorted(per.unique()))
    tindex = {p: i for i, p in enumerate(periods)}
    it = per.map(tindex).values
    T = len(periods)
    fname = {"D": "daily", "W": "weekly", "M": "monthly"}[args.freq]
    print(f"grid {G}x{G} cells | {T} time steps ({fname})")
    if int(T * .70) < 100:
        print(f"  WARNING: only {int(T*.70)} training time steps. A PDE needs a"
              f"\n  time derivative -- aim for 100+. Widen --year0/--year1 or use --freq D.")

    # ---- counts ----------------------------------------------------------- #
    U = np.zeros((T, G, G), dtype=np.float64)
    np.add.at(U, (it, iy, ix), 1.0)

    # ---- mask: cells that never see a crime are outside the city ---------- #
    total = U.sum(0)
    mask = total >= args.min_events
    print(f"in-city cells: {mask.sum()} of {G*G} "
          f"({100*mask.sum()/(G*G):.1f}%)  [min_events={args.min_events}]")

    # ---- community area per cell (modal) ---------------------------------- #
    area = np.full((G, G), -1, dtype=np.int32)
    ca = pd.to_numeric(df["community_area"], errors="coerce")
    ok = ca.notna().values
    tmp = pd.DataFrame({"iy": iy[ok], "ix": ix[ok], "ca": ca[ok].astype(int).values})
    modal = tmp.groupby(["iy", "ix"])["ca"].agg(lambda s: s.value_counts().idxmax())
    for (yy, xx), v in modal.items():
        area[yy, xx] = v

    # ---- Head / Mid / Tail per cell, by total crime (20/30/50) ------------ #
    group = np.full((G, G), "", dtype=object)
    idx = np.argwhere(mask)
    vals = total[mask]
    order = np.argsort(vals)[::-1]
    n = len(order); nh = max(1, round(n * .20)); nm = max(1, round(n * .30))
    for rank, o in enumerate(order):
        yy, xx = idx[o]
        group[yy, xx] = "Head" if rank < nh else ("Mid" if rank < nh + nm else "Tail")

    # ---- smoothing -------------------------------------------------------- #
    U_raw = U.copy()
    U = gaussian_blur(U, args.smooth)
    U *= mask[None, :, :]

    # ---- normalised coordinates ------------------------------------------- #
    latc = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lonc = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    yy, xx = np.meshgrid(latc, lonc, indexing="ij")
    xn = 2 * (xx - LON_LO) / (LON_HI - LON_LO) - 1
    yn = 2 * (yy - LAT_LO) / (LAT_HI - LAT_LO) - 1
    tn = np.linspace(0.0, 1.0, T)

    # ---- time split (never random) ---------------------------------------- #
    ntr, nva = int(T * .70), int(T * .10)
    split = np.array(["train"] * ntr + ["val"] * nva + ["test"] * (T - ntr - nva))

    # ---- report ----------------------------------------------------------- #
    inside = U[:, mask]
    print(f"\nfield summary")
    print(f"  U shape           {U.shape}")
    print(f"  mean intensity    {inside.mean():.4f} events/cell/step")
    print(f"  max  intensity    {inside.max():.2f}")
    print(f"  empty cell-steps  {100*(U_raw[:, mask] == 0).mean():.1f}% "
          f"(before smoothing)")
    for g in ("Head", "Mid", "Tail"):
        m = (group == g)
        print(f"  {g:5s} {m.sum():4d} cells   mean {U[:, m].mean():.4f}")
    print(f"  split             train {ntr} / val {nva} / test {T-ntr-nva} steps")

    np.savez_compressed(
        args.out, U=U.astype(np.float32), U_raw=U_raw.astype(np.float32),
        x=xn.astype(np.float32), y=yn.astype(np.float32), t=tn.astype(np.float32),
        mask=mask, area=area, group=group.astype("U4"), split=split,
        lat_edges=lat_edges, lon_edges=lon_edges,
        meta=json.dumps({"years": [args.year0, args.year1], "grid": G, "freq": args.freq,
                         "smooth": args.smooth, "min_events": args.min_events,
                         "category": args.category, "n_records": int(len(df)),
                         "domain": {"x": [-1, 1], "y": [-1, 1], "t": [0, 1]}}))
    print(f"\nsaved -> {args.out}")
    print("domain for PDE collocation points: x,y in [-1,1], t in [0,1]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year0", type=int, default=2015, help="first year (inclusive)")
    ap.add_argument("--year1", type=int, default=None, help="last year; default = year0")
    ap.add_argument("--grid", type=int, default=48, help="cells per side")
    ap.add_argument("--freq", default="D", choices=["D", "W", "M"])
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="Gaussian sigma in cells; 0 = none")
    ap.add_argument("--min-events", type=int, default=5,
                    help="a cell needs this many events all year to count as in-city")
    ap.add_argument("--category", default=None,
                    help="e.g. BURGLARY - the Short et al. model is a burglary model")
    ap.add_argument("--csv", default=None, help="use a local CSV instead of the API")
    ap.add_argument("--out", default="data/chi_field.npz")
    args = ap.parse_args()
    if args.year1 is None:
        args.year1 = args.year0
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    build(args)


if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules:
        print("=" * 70)
        print("This file was RUN as a notebook cell instead of being SAVED.")
        print("Add this as the FIRST line of this cell, then press Enter:")
        print()
        print("    %%writefile build_field.py")
        print()
        print("Then run it with:  !python build_field.py --year 2015 ...")
        print("=" * 70)
    else:
        main()
