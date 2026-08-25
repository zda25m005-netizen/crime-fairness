# Reproducing everything

---

## 0. Install

```bash
pip install -r requirements.txt
```

GPU work needs PyTorch with CUDA. The notebooks in `notebooks/` install
everything themselves and are the easiest route.

---

## 1. No GPU, no data — run these first (seconds each)

```bash
python analysis/theory.py                # the maths, verified numerically
python analysis/literature_audit.py      # 6 published models vs crime volume
python analysis/sensitivity.py           # W2/W3: assumptions and small-n
python analysis/decision_impact.py       # does it change decisions?
python analysis/reanalysis_published.py  # FedCrime's own tables
```

These reproduce Sections 1–4 of [`RESULTS.md`](RESULTS.md) exactly. They depend
only on published numbers and NumPy.

---

## 2. Build the datasets

**Los Angeles** — ships inside the FedCrime repository:

```bash
mkdir -p data
git clone --depth 1 https://github.com/vanetlabiitj/FedCrime.git
cp FedCrime/Dataset/processed_crime.csv data/la_crime.csv
# expect 182,325 rows
```

**Chicago, New York, San Francisco, Chicago 311** — from the open-data portals:

```bash
python scripts/preprocess_chicago.py --out data/chi_crime.csv
python scripts/preprocess_cities.py --city nyc    --year 2019 --out data/nyc_crime.csv
python scripts/preprocess_cities.py --city sf     --year 2019 --out data/sf_crime.csv
python scripts/preprocess_cities.py --city chi311 --year 2019 --out data/chi311.csv
```

**The burglary field for the PINN** (point records → continuous field):

```bash
python src/pinn/build_field.py --year0 2011 --year1 2015 --freq W --grid 24 \
    --category BURGLARY --smooth 1.2 --out data/burg_w24.npz
# expect 94,992 records, 262 weekly steps, 183 training weeks, ~41.5% empty
```

---

## 3. The reproducibility gate — always run this first

```bash
python src/fedcrime/robust_fair_gnn.py --city chicago --repro-check \
    --gnn plain --gnn-layers 2 --rounds 20 --seed 0
```

Must print **PASS** with difference 0.00000000. If it prints FAIL, stop —
nothing downstream can be trusted.

---

## 4. Graph depth ablation (Section 5)

```bash
for g in plain gated attention; do
  python src/fedcrime/robust_fair_gnn.py --city la --depth-sweep --gnn $g \
      --seeds 5 --rounds 40 --save results/depth_paired_la.jsonl
done
```

Repeat with `--city chicago`. ~1 hour per city on a T4.

---

## 5. Threshold sweep (Section 6) — the key experiment

```bash
python analysis/threshold_sweep.py --city chicago --seeds 5 --rounds 40 \
    --target-f1 40.38 --out results/threshold_sweep.json
```

Watch two things: Tail AUC must be **constant** down the whole table, and the
best threshold-only Tail F1 should reach or exceed the graph model's 40.38.

---

## 6. PINN (Sections 7–8)

```bash
# is there week-to-week signal at all?
python src/pinn/signal_check.py --data data/burg_w24.npz

# 15 runs: physics / physics+MSE / control, 5 seeds each  (~45 min)
for s in 0 1 2 3 4; do
  python src/pinn/crime_pinn.py --data data/burg_w24.npz --physics short --loss poisson --seed $s --save results/pinn.jsonl
  python src/pinn/crime_pinn.py --data data/burg_w24.npz --physics short --loss mse     --seed $s --save results/pinn.jsonl
  python src/pinn/crime_pinn.py --data data/burg_w24.npz --physics none  --loss poisson --seed $s --save results/pinn.jsonl
done

# the decisive ablation
python src/pinn/crime_pinn.py --data data/burg_w24.npz --physics smooth --loss poisson

# final table with mean, std and Welch t-tests
python analysis/analyse_pinn.py --file results/pinn.jsonl
```

---

## Notebooks

If you would rather not manage the environment, these are self-contained:

| Notebook | What it does |
|---|---|
| `notebooks/Crime_PINN_Drive.ipynb` | Full PINN pipeline, saves to Google Drive so a disconnect costs nothing |
| `notebooks/FedCrime_Reviewer_Response.ipynb` | Sensitivity analysis + threshold sweep, Kaggle or Colab |
| `notebooks/FedCrime_Chicago_Replication.ipynb` | Chicago graph depth sweep |

---

## Hardware note

All reported numbers come from a single NVIDIA T4. Runs are bit-identical when
repeated on the same hardware. Across different GPUs, direction and statistical
significance are stable, but exact values vary in the third digit.
