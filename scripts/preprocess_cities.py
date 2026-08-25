"""
Build FedCrime-format datasets for additional cities from open-data portals.

Produces ``data/<city>_crime.csv`` with the schema the pipeline expects:

    date_occ, crime_type_id, neighborhood_id

Each city is mapped onto EIGHT crime categories so results are comparable
across cities (as in the FedCrime paper's LA/Chicago setup).

Supported
---------
  nyc   New York City  — 77 police precincts   (Socrata qgea-i56i)
  sf    San Francisco  — ~41 analysis neighbourhoods (Socrata wg3w-h783)
  phl   Philadelphia   — 21 police districts   (Carto SQL API)

Usage
-----
    python scripts/preprocess_cities.py --city nyc --year 2019 --out data/nyc_crime.csv
    python scripts/preprocess_cities.py --city sf  --year 2019 --out data/sf_crime.csv
    python scripts/preprocess_cities.py --city phl --year 2019 --out data/phl_crime.csv

Requires network access on the machine you run this on.
"""
from __future__ import annotations

import argparse
import time
from urllib.parse import urlencode

import pandas as pd

# --------------------------------------------------------------------------- #
# City configurations: 8 categories each, mapped from the portal's own labels
# --------------------------------------------------------------------------- #
NYC = {
    "resource": "https://data.cityofnewyork.us/resource/qgea-i56i.json",
    "date": "cmplnt_fr_dt", "cat": "ofns_desc", "region": "addr_pct_cd",
    # 0 theft 1 assault 2 criminal-mischief 3 harassment
    # 4 robbery 5 burglary 6 narcotics 7 sex-crimes
    "map": {
        "PETIT LARCENY": 0, "GRAND LARCENY": 0,
        "ASSAULT 3 & RELATED OFFENSES": 1, "FELONY ASSAULT": 1,
        "CRIMINAL MISCHIEF & RELATED OF": 2,
        "HARRASSMENT 2": 3,
        "ROBBERY": 4,
        "BURGLARY": 5,
        "DANGEROUS DRUGS": 6,
        "SEX CRIMES": 7,
    },
    "names": ["theft", "assault", "criminal_mischief", "harassment",
              "robbery", "burglary", "narcotics", "sex_crimes"],
}

SF = {
    "resource": "https://data.sfgov.org/resource/wg3w-h783.json",
    "date": "incident_date", "cat": "incident_category",
    "region": "analysis_neighborhood",
    "map": {
        "Larceny Theft": 0,
        "Assault": 1,
        "Malicious Mischief": 2,
        "Other Miscellaneous": 3,
        "Robbery": 4,
        "Burglary": 5,
        "Drug Offense": 6,
        "Motor Vehicle Theft": 7,
    },
    "names": ["theft", "assault", "malicious_mischief", "other",
              "robbery", "burglary", "narcotics", "vehicle_theft"],
}

# SECOND DOMAIN (not crime): Chicago 311 service requests, 2019.
# Same 77 community areas as the Chicago crime data, so the spatial units and
# the Head/Mid/Tail construction are identical -- this isolates the DOMAIN as
# the only thing that changes.  Categories are chosen automatically as the eight
# most frequent request types (auto_top8), which avoids hard-coding a taxonomy.
CHI311 = {
    "resource": "https://data.cityofchicago.org/resource/v6vf-nfxy.json",
    "date": "created_date", "cat": "sr_type", "region": "community_area",
    "map": None,               # None -> take the 8 most frequent categories
    "names": None,
}

CITIES = {"nyc": NYC, "sf": SF, "chi311": CHI311}


def fetch_socrata(cfg, year, page=50000, max_rows=1_500_000):
    """Page a Socrata endpoint for one calendar year."""
    rows, offset = [], 0
    date, cat, region = cfg["date"], cfg["cat"], cfg["region"]
    where = (f"{date} >= '{year}-01-01T00:00:00.000' "
             f"AND {date} <= '{year}-12-31T23:59:59.000'")
    while offset < max_rows:
        q = urlencode({"$select": f"{date},{cat},{region}",
                       "$where": where, "$limit": page, "$offset": offset})
        chunk = pd.read_json(f"{cfg['resource']}?{q}")
        if chunk.empty:
            break
        rows.append(chunk)
        offset += page
        print(f"  fetched {offset} rows...")
        time.sleep(0.4)
    if not rows:
        raise SystemExit("No rows returned — check the year or the portal.")
    return pd.concat(rows, ignore_index=True)


def build(city: str, year: int, out: str, local_csv: str | None = None):
    cfg = CITIES[city]
    date, cat, region = cfg["date"], cfg["cat"], cfg["region"]

    df = pd.read_csv(local_csv) if local_csv else fetch_socrata(cfg, year)
    df = df.dropna(subset=[date, cat, region])

    mapping = cfg["map"]
    if mapping is None:                    # auto: eight most frequent categories
        top8 = df[cat].astype(str).str.strip().value_counts().head(8).index.tolist()
        mapping = {name: i for i, name in enumerate(top8)}
        cfg = dict(cfg, names=[t[:20] for t in top8])
        print("auto-selected categories:")
        for i, t in enumerate(top8):
            print(f"  {i} {t}")

    # map the portal's own labels onto the 8 shared categories
    df["crime_type_id"] = df[cat].astype(str).str.strip().map(mapping)
    df = df.dropna(subset=["crime_type_id"])
    df["crime_type_id"] = df["crime_type_id"].astype(int)

    # regions -> contiguous integer ids
    codes, uniques = pd.factorize(df[region].astype(str).str.strip())
    df["neighborhood_id"] = codes
    df = df[df["neighborhood_id"] >= 0]

    df["date_occ"] = pd.to_datetime(df[date]).dt.strftime("%Y-%m-%d")

    outdf = df[["date_occ", "crime_type_id", "neighborhood_id"]].copy()
    outdf.to_csv(out, index=False)

    print(f"\nWrote {len(outdf)} rows to {out}")
    print(f"regions: {outdf.neighborhood_id.nunique()} | "
          f"categories: {sorted(outdf.crime_type_id.unique())}")
    counts = outdf.crime_type_id.value_counts().sort_index()
    for i, n in counts.items():
        print(f"  {i} {cfg['names'][i]:20s} {n}")
    # sparsity preview (daily presence per region/category)
    days = pd.to_datetime(outdf.date_occ).nunique()
    R = outdf.neighborhood_id.nunique()
    cells = days * R * 8
    present = outdf.drop_duplicates(["date_occ", "neighborhood_id",
                                     "crime_type_id"]).shape[0]
    print(f"\napprox label sparsity: {100*(1-present/cells):.1f}% zeros "
          f"({days} days x {R} regions x 8 categories)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(CITIES))
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None,
                    help="use a locally downloaded CSV instead of the API")
    args = ap.parse_args()
    build(args.city, args.year, args.out, args.csv)


if __name__ == "__main__":
    main()
