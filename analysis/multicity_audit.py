r"""
Multi-city replication of the MEASUREMENT CRITIQUE  (FAccT item 1)
================================================================================
WHAT THIS REPLICATES — and what it does NOT
-------------------------------------------
It does NOT train any model. There is nothing to replicate about our physics
model: it did not work.

It replicates the CRITIQUE, which needs no model at all:

  (a) In every city, splitting regions into Head/Mid/Tail by crime volume
      produces groups with very different base rates.
  (b) A model with EXACTLY EQUAL SKILL in every group still shows a large
      Head-Tail F1 gap, purely from those base rates.
  (c) Pooling AUC across regions inflates it, because a scorer can win by
      ranking REGIONS instead of ranking WEEKS. Macro (within-region) AUC
      does not inflate.

Because (b) and (c) are simulated with a scorer we construct to have equal
skill everywhere, we know the ground truth: the honest gap is ZERO. Anything
the metric reports is artifact. Running this across 6-8 cities turns a
one-city observation into a general property of the data, and it costs
minutes rather than GPU-days.

USAGE
-----
    # 1. check the endpoints respond and have the fields we expect
    python multicity_audit.py --probe

    # 2. run the full audit
    python multicity_audit.py --year0 2015 --year1 2019 --grid 24

    # offline: put CSVs with columns lat,lon,date in ./citycsv/<name>.csv
    python multicity_audit.py --csv-dir citycsv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd

GROUPS = ("Head", "Mid", "Tail")

# --------------------------------------------------------------------------- #
# City endpoints. Socrata portals share a query language but NOT field names,
# so each city carries its own mapping. bbox trims mis-geocoded rows (0,0 is a
# common junk value) and water.
# --------------------------------------------------------------------------- #
CITIES = {
    "Chicago": dict(
        url="https://data.cityofchicago.org/resource/ijzp-q8t2.json",
        date="date", lat="latitude", lon="longitude", cat="primary_type",
        bbox=(41.644, 42.023, -87.940, -87.524)),
    "Los Angeles": dict(
        url="https://data.lacity.org/resource/2nrs-mtv8.json",
        date="date_occ", lat="lat", lon="lon", cat="crm_cd_desc",
        y0=2020, y1=2024,
        bbox=(33.70, 34.34, -118.67, -118.15)),
    "New York": dict(
        url="https://data.cityofnewyork.us/resource/qgea-i56i.json",
        date="cmplnt_fr_dt", lat="latitude", lon="longitude", cat="ofns_desc",
        bbox=(40.49, 40.92, -74.26, -73.70)),
    "Seattle": dict(
        url="https://data.seattle.gov/resource/tazs-3rd5.json",
        date="offense_date", lat="latitude", lon="longitude",
        cat="offense_sub_category", bbox=(47.48, 47.74, -122.44, -122.22)),
    "Cincinnati": dict(
        url="https://data.cincinnati-oh.gov/resource/k59e-2pvf.json",
        date="date_reported", lat="latitude_x", lon="longitude_x",
        cat="offense", bbox=(39.05, 39.23, -84.72, -84.36)),
}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auc_np(y, s):
    y = np.asarray(y).ravel().astype(int); s = np.asarray(s, float).ravel()
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    _, f, c = np.unique(s[o], return_index=True, return_counts=True)
    for st, ct in zip(f[c > 1], c[c > 1]):
        i = o[st:st + ct]; r[i] = r[i].mean()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def macro_auc(y, s, unit, min_pos=3):
    """Mean of per-unit AUCs. Ranking units against each other cannot help."""
    v = []
    for u in np.unique(unit):
        m = unit == u; yy = y[m]; np_ = int(yy.sum())
        if np_ < min_pos or (len(yy) - np_) < min_pos:
            continue
        v.append(auc_np(yy, s[m]))
    return (100 * float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def f1_at(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return 0.0 if 2*tp+fp+fn == 0 else 100 * 2*tp / (2*tp+fp+fn)


def best_f1(y, s, n=120):
    grid = np.quantile(s, np.linspace(0.02, 0.98, n))
    return max(f1_at(y, (s > t).astype(int)) for t in grid)


def a_f1(p):
    """F1 of the skill-free predictor that says YES everywhere: 2p/(1+p)."""
    return 100 * 2 * p / (1 + p)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
UA = {"User-Agent": "crime-fairness-audit/1.0 (academic research)",
      "Accept": "application/json"}


def _read_json(url, timeout=180, tries=3):
    # Hard timeout (pandas gives urllib none), a User-Agent (some portals
    # reject clients without one), and retries (portals 5xx under load).
    import io, requests
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers=UA)
            r.raise_for_status()
            return pd.read_json(io.StringIO(r.text))
        except Exception as e:
            last = e
            time.sleep(2 ** k)
    raise last


def probe(name, cfg, timeout=30):
    """Fetch 3 rows and report whether the fields we need are present."""
    try:
        q = urlencode({"$limit": 200})
        df = _read_json(f"{cfg['url']}?{q}")
    except Exception as e:
        return f"FETCH FAILED: {type(e).__name__}: {str(e)[:70]}"
    have = set(df.columns)
    need = {cfg["date"], cfg["lat"], cfg["lon"]}
    missing = need - have
    if missing:
        return (f"MISSING {sorted(missing)}\n      available: "
                f"{sorted(have)[:12]}")
    return f"OK  ({len(df.columns)} columns)"


def fetch_city(name, cfg, year0, year1, category=None, page=50_000,
               max_rows=1_500_000):
    # Socrata re-runs the whole filtered scan for every page, so each
    # successive $offset is slower than the last -- Chicago and New York both
    # timed out on page two. One query per YEAR keeps every request small.
    # A year that fails is skipped, not fatal: this audit measures the SHAPE
    # of the base-rate distribution, which four years shows as well as five.
    y0 = cfg.get("y0", year0)
    y1 = cfg.get("y1", year1)
    frames, total = [], 0
    for yr in range(y0, y1 + 1):
        where = (f"{cfg['date']} >= '{yr}-01-01T00:00:00.000' "
                 f"AND {cfg['date']} <= '{yr}-12-31T23:59:59.000' "
                 f"AND {cfg['lat']} IS NOT NULL")
        if category and cfg.get("cat"):
            where += f" AND upper({cfg['cat']}) like '%{category.upper()}%'"
        off = 0
        while off < max_rows:
            q = urlencode({"$select": f"{cfg['date']},{cfg['lat']},{cfg['lon']}",
                           "$where": where, "$order": cfg["date"],
                           "$limit": page, "$offset": off})
            try:
                ch = _read_json(f"{cfg['url']}?{q}")
            except Exception as e:
                print(f"      {yr}: FAILED {type(e).__name__} - year skipped",
                      flush=True)
                break
            if ch.empty:
                break
            frames.append(ch)
            off += page
            total += len(ch)
            if len(ch) < page:
                break
            time.sleep(0.3)
        print(f"      {yr}: {total:,} rows so far", flush=True)
    if not frames:
        raise RuntimeError("no rows")
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={cfg["date"]: "date", cfg["lat"]: "lat",
                              cfg["lon"]: "lon"})


def gridify(df, bbox, grid, freq="W", min_events=5):
    """Point records -> (weeks, cells) count matrix, plus Head/Mid/Tail split."""
    la0, la1, lo0, lo1 = bbox
    df = df.dropna(subset=["lat", "lon", "date"]).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["lat", "lon", "date"])
    df = df[(df.lat.between(la0, la1)) & (df.lon.between(lo0, lo1))]
    if len(df) < 5000:
        raise RuntimeError(f"only {len(df)} usable rows after cleaning")

    yi = np.clip(((df.lat - la0) / (la1 - la0) * grid).astype(int), 0, grid - 1)
    xi = np.clip(((df.lon - lo0) / (lo1 - lo0) * grid).astype(int), 0, grid - 1)
    cell = (yi * grid + xi).to_numpy()
    per = df["date"].dt.to_period(freq)
    tcode = (per - per.min()).apply(lambda x: x.n).to_numpy()
    T = int(tcode.max()) + 1

    M = np.zeros((T, grid * grid), np.float64)
    np.add.at(M, (tcode, cell), 1.0)
    keep = M.sum(0) >= min_events              # in-city cells only
    M = M[:, keep]
    vol = M.sum(0)
    order = np.argsort(-vol)                   # busiest first
    n = len(vol)
    grp = np.empty(n, dtype="U4")
    grp[order[:int(.20*n)]] = "Head"
    grp[order[int(.20*n):int(.50*n)]] = "Mid"
    grp[order[int(.50*n):]] = "Tail"
    return M, grp, len(df)


# --------------------------------------------------------------------------- #
# the audit: simulate a model with EQUAL SKILL everywhere
# --------------------------------------------------------------------------- #
def audit(M, grp, rng, d_prime=1.0):
    """M is (weeks, cells) counts. Build labels y = (count>0), then a scorer
    whose signal strength is IDENTICAL in every cell:

        score = d_prime * y + noise,   noise ~ N(0,1)

    Because the same d_prime and the same noise scale are used everywhere,
    every cell -- and therefore every group -- has the SAME ROC curve. The
    honest Head-Tail gap is exactly zero. Whatever a metric reports is
    artifact, and we can measure how much.
    """
    T, n = M.shape
    y = (M > 0).astype(int)
    s = d_prime * y + rng.normal(0, 1, y.shape)

    unit = np.tile(np.arange(n), T)
    yf, sf, gf = y.reshape(-1), s.reshape(-1), np.tile(grp, T)

    out = {}
    for k in GROUPS:
        m = gf == k
        out[k] = dict(
            base=100 * yf[m].mean(),
            af1=a_f1(yf[m].mean()),                 # skill-free F1
            f1=best_f1(yf[m], sf[m]),               # equal-skill model's F1
            auc_pool=100 * auc_np(yf[m], sf[m]),
            auc_macro=macro_auc(yf[m], sf[m], unit[m])[0])
    out["gap_base"] = out["Head"]["base"] - out["Tail"]["base"]
    out["gap_af1"] = out["Head"]["af1"] - out["Tail"]["af1"]
    out["gap_f1"] = out["Head"]["f1"] - out["Tail"]["f1"]
    out["gap_auc_pool"] = out["Head"]["auc_pool"] - out["Tail"]["auc_pool"]
    out["gap_auc_macro"] = out["Head"]["auc_macro"] - out["Tail"]["auc_macro"]
    out["auc_inflation"] = (100 * auc_np(yf, sf)) - macro_auc(yf, sf, unit)[0]

    # ------------------------------------------------------------------ #
    # SECOND EXPERIMENT: what happens when the scorer ALSO knows which
    # cell it is looking at? A trained model learns this for free -- it is
    # exactly what inflated our own headline result by ~10 AUC points.
    # beta controls how much the scorer leans on cell identity. Skill
    # WITHIN each cell is unchanged at every beta, so macro AUC must stay
    # flat while pooled AUC drifts.
    # ------------------------------------------------------------------ #
    cell_rate = y.mean(0)                     # each cell's own base rate
    sweep = []
    for beta in (0.0, 1.0, 2.0, 4.0):
        s2 = (d_prime * y + rng.normal(0, 1, y.shape)
              + beta * cell_rate[None, :]).reshape(-1)
        sweep.append(dict(beta=beta,
                          pooled=100 * auc_np(yf, s2),
                          macro=macro_auc(yf, s2, unit)[0]))
    out["sweep"] = sweep
    out["max_inflation"] = max(x["pooled"] - x["macro"] for x in sweep)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="just check the endpoints and field names")
    ap.add_argument("--year0", type=int, default=2015)
    ap.add_argument("--year1", type=int, default=2019)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--category", default=None,
                    help="e.g. BURGLARY; omit for all crime")
    ap.add_argument("--min-events", type=int, default=5)
    ap.add_argument("--d-prime", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated city names")
    ap.add_argument("--save", default="results/multicity_audit.json")
    args = ap.parse_args()

    cities = CITIES
    if args.only:
        want = {c.strip().lower() for c in args.only.split(",")}
        cities = {k: v for k, v in CITIES.items() if k.lower() in want}

    if args.probe:
        print("=" * 74)
        print("PROBE — do the endpoints respond, and do the fields exist?")
        print("=" * 74)
        for name, cfg in cities.items():
            print(f"  {name:>15}: {probe(name, cfg)}")
        print("\n  Fix or drop any city that is not OK before the full run.")
        return

    rng = np.random.default_rng(args.seed)
    rows = []
    print("=" * 74)
    print("MULTI-CITY AUDIT — equal-skill model, measured by each metric")
    print("=" * 74)
    print(f"  years {args.year0}-{args.year1}   grid {args.grid}x{args.grid}   "
          f"category {args.category or 'ALL'}   d' {args.d_prime}")

    for name, cfg in cities.items():
        print(f"\n  --- {name} ---")
        try:
            if args.csv_dir:
                p = os.path.join(args.csv_dir, f"{name.replace(' ','_')}.csv")
                df = pd.read_csv(p)
            else:
                df = fetch_city(name, cfg, args.year0, args.year1, args.category)
            M, grp, nrec = gridify(df, cfg["bbox"], args.grid,
                                   min_events=args.min_events)
        except Exception as e:
            print(f"      SKIPPED: {type(e).__name__}: {str(e)[:80]}")
            continue
        r = audit(M, grp, rng, args.d_prime)
        r.update(city=name, records=nrec, weeks=M.shape[0], cells=M.shape[1])
        rows.append(r)
        print(f"      {nrec:,} records   {M.shape[0]} weeks   {M.shape[1]} cells")
        print(f"      base rate  Head {r['Head']['base']:.1f}%  "
              f"Mid {r['Mid']['base']:.1f}%  Tail {r['Tail']['base']:.1f}%")

    if not rows:
        print("\n  No cities succeeded. Run --probe to see why.")
        return

    # ---------------- the table that goes in the paper ------------------- #
    print("\n" + "=" * 74)
    print("RESULT — a model with IDENTICAL skill in every group")
    print("        (so the honest Head-Tail gap is ZERO in every column)")
    print("=" * 74)
    print(f"  {'city':>14} {'base gap':>9} {'F1 gap':>8} {'pooled AUC gap':>15} "
          f"{'macro AUC gap':>14}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['city']:>14} {r['gap_base']:+8.1f}% {r['gap_f1']:+8.2f} "
              f"{r['gap_auc_pool']:+15.2f} {r['gap_auc_macro']:+14.2f}")
    f1g = np.array([r["gap_f1"] for r in rows])
    apg = np.array([r["gap_auc_pool"] for r in rows])
    amg = np.array([r["gap_auc_macro"] for r in rows])
    print("  " + "-" * 70)
    print(f"  {'MEAN':>14} {np.mean([r['gap_base'] for r in rows]):+8.1f}% "
          f"{f1g.mean():+8.2f} {apg.mean():+15.2f} {amg.mean():+14.2f}")
    print(f"  {'SD':>14} {'':9} {f1g.std(ddof=1):8.2f} "
          f"{apg.std(ddof=1):15.2f} {amg.std(ddof=1):14.2f}")

    print("\n" + "=" * 74)
    print("POOLING INFLATION — scorer leans increasingly on CELL IDENTITY")
    print("  (within-cell skill is identical at every beta, so macro must be flat)")
    print("=" * 74)
    betas = [x["beta"] for x in rows[0]["sweep"]]
    print(f"  {'city':>14} " + "".join(f"{'b='+str(b):>17}" for b in betas))
    print(f"  {'':>14} " + "".join(f"{'pooled / macro':>17}" for _ in betas))
    print("  " + "-" * (14 + 17 * len(betas)))
    for r in rows:
        cells = "".join(f"{x['pooled']:7.1f} /{x['macro']:6.1f}  "
                        for x in r["sweep"])
        print(f"  {r['city']:>14} {cells}")
    inf = np.array([r["max_inflation"] for r in rows])
    print(f"\n  worst pooled-minus-macro inflation per city: "
          f"mean {inf.mean():+.2f}, max {inf.max():+.2f}")
    mac = np.array([[x["macro"] for x in r["sweep"]] for r in rows])
    print(f"  macro AUC drift across beta (should be ~0): "
          f"{np.abs(mac - mac[:, :1]).max():.2f} points worst case")

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print(f"""  Every number above comes from a model built to be EQUALLY GOOD
  in every group. The truthful answer for all of them is 0.00.

    F1 gap          {f1g.mean():+6.2f} on average  -> the artifact, in {len(rows)} cities
    macro AUC gap   {amg.mean():+6.2f} on average  -> correct, as it must be
    pooling inflation up to {inf.max():+5.2f}     -> what pooling can add for free

  If the F1 column is large and the macro-AUC column is near zero in every
  city, the critique is no longer a one-city observation. It is a property
  of how this data is shaped, and it will hold anywhere regions differ in
  event rate -- which is everywhere.""")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        json.dump(rows, open(args.save, "w"), indent=2, default=float)
        print(f"\n  saved -> {args.save}")


if __name__ == "__main__":
    main()
