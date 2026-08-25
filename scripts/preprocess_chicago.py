"""
Build the Chicago (2015) FedCrime dataset from the City of Chicago open-data
portal, producing ``data/chi_crime.csv`` with the schema the pipeline expects:

    date_occ, crime_type_id, neighborhood_id

The eight crime categories and their ids follow the paper's Chicago ordering:

    0 robbery   1 battery    2 deceptive_practice  3 burglary
    4 assault   5 theft      6 criminal_damage     7 narcotics

Two input modes
---------------
1. Direct API (default): pages the Socrata endpoint for the eight categories in
   2015. Requires network access on the machine you run this on.

       python scripts/preprocess_chicago.py --out data/chi_crime.csv

2. Local CSV: if you've downloaded the full "Crimes - 2001 to Present" CSV
   (columns include ``Date``, ``Primary Type``, ``Community Area``, ``Year``),
   point at it and skip the network:

       python scripts/preprocess_chicago.py --csv Crimes_-_2001_to_Present.csv \
           --out data/chi_crime.csv

Category counts should closely match the paper (theft ~57k, battery ~49k,
criminal damage ~29k, narcotics ~24k, assault ~17k, deceptive ~16k,
burglary ~13k, robbery ~10k).
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

# paper Chicago primary_type -> crime_type_id
PRIMARY_TO_ID = {
    "ROBBERY": 0,
    "BATTERY": 1,
    "DECEPTIVE PRACTICE": 2,
    "BURGLARY": 3,
    "ASSAULT": 4,
    "THEFT": 5,
    "CRIMINAL DAMAGE": 6,
    "NARCOTICS": 7,
}
RESOURCE = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"


def from_api(year: int = 2015) -> pd.DataFrame:
    from urllib.parse import urlencode
    types = "','".join(PRIMARY_TO_ID)
    where = f"year={year} AND primary_type IN('{types}')"
    rows, offset, page = [], 0, 50000
    while True:
        # URL-encode the query so spaces/quotes in $where don't break the URL
        q = urlencode({"$select": "date,primary_type,community_area",
                       "$where": where, "$limit": page, "$offset": offset})
        url = f"{RESOURCE}?{q}"
        chunk = pd.read_json(url)
        if chunk.empty:
            break
        rows.append(chunk)
        offset += page
        print(f"  fetched {offset} rows...")
        time.sleep(0.5)
    return pd.concat(rows, ignore_index=True)


def from_csv(path: str, year: int = 2015) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Date", "Primary Type",
                                    "Community Area", "Year"])
    df = df[df["Year"] == year]
    df = df[df["Primary Type"].isin(PRIMARY_TO_ID)]
    return df.rename(columns={"Date": "date", "Primary Type": "primary_type",
                              "Community Area": "community_area"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="local Chicago crimes CSV")
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--out", default="data/chi_crime.csv")
    args = ap.parse_args()

    df = from_csv(args.csv, args.year) if args.csv else from_api(args.year)

    df = df.dropna(subset=["community_area", "primary_type", "date"])
    df["crime_type_id"] = df["primary_type"].str.upper().map(PRIMARY_TO_ID)
    df = df.dropna(subset=["crime_type_id"])
    df["crime_type_id"] = df["crime_type_id"].astype(int)
    # community areas 1..77 -> neighborhood_id 0..76
    df["neighborhood_id"] = (pd.to_numeric(df["community_area"], errors="coerce")
                             .astype("Int64") - 1)
    df = df.dropna(subset=["neighborhood_id"])
    df["date_occ"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    out = df[["date_occ", "crime_type_id", "neighborhood_id"]].copy()
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows to {args.out}")
    print(f"regions: {out.neighborhood_id.nunique()}  "
          f"categories: {sorted(out.crime_type_id.unique())}")
    print(out.crime_type_id.value_counts().sort_index())


if __name__ == "__main__":
    main()
