"""
Lightweight self-tests for FedCrime.

Run with:  python -m pytest tests/  (or)  python tests/test_components.py

The data-pipeline and ZINB-math tests run WITHOUT PyTorch. The model /
training tests are skipped automatically if torch is not installed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# data.py has no torch dependency, import it directly
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "fc_data", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "fedcrime", "data.py"))
data = importlib.util.module_from_spec(_spec)
sys.modules["fc_data"] = data
_spec.loader.exec_module(data)

try:
    import torch  # noqa
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "la_crime.csv")


def test_data_pipeline():
    regions = data.build_regions(DATA)
    assert len(regions) == 113, "LA dataset should yield 113 regions"
    cats = data.categorize_regions(regions)
    assert len(cats["H"]) + len(cats["M"]) + len(cats["T"]) == 113
    rid = next(iter(regions))
    assert regions[rid].x_train.shape[1:] == (data.WINDOW - 2, data.NUM_CATEGORIES)
    print("[ok] data pipeline: 113 regions, window shape",
          regions[rid].x_train.shape[1:])


def test_subset_sparsity_ordering():
    regions = data.build_regions(DATA)
    sps = [data.sparsity(regions, data.build_subset(regions, s))
           for s in ("alpha", "beta", "gamma", "omega")]
    assert all(sps[i] < sps[i + 1] for i in range(3)), \
        f"sparsity must increase across subsets, got {sps}"
    print("[ok] subset sparsity increases:",
          [round(s, 1) for s in sps])


def test_zinb_math():
    import math
    lgamma = np.vectorize(math.lgamma)
    eps = 1e-10

    def zinb(pi, mu, phi, y):
        is0 = (y == 0).astype(float)
        is1 = (y > 0).astype(float)
        zero = is0 * np.log(pi + eps)
        nb = is1 * (lgamma(y + phi) - lgamma(phi) - lgamma(y + 1)
                    + phi * (np.log(phi + eps) - np.log(phi + mu + eps))
                    + y * (np.log(mu + eps) - np.log(phi + mu + eps)))
        return -np.mean(zero + nb)

    mu = np.ones((1, 2)); phi = np.ones((1, 2)); y0 = np.zeros((1, 2))
    assert abs(zinb(np.ones((1, 2)), mu, phi, y0)) < 1e-6      # perfect zeros
    assert zinb(np.full((1, 2), 0.3), mu, phi, y0) > \
           zinb(np.full((1, 2), 0.9), mu, phi, y0)             # monotone in pi
    print("[ok] ZINB loss math validated")


def test_model_and_train():
    if not HAS_TORCH:
        print("[skip] torch not installed -> model/train tests skipped")
        return
    from fedcrime.model import TCN_SD
    from fedcrime.federated import train_federated, evaluate_federated
    regions = data.build_regions(DATA)
    ids = data.build_subset(regions, "alpha")
    net = TCN_SD()
    x = torch.zeros((4, data.WINDOW - 2, data.NUM_CATEGORIES))
    pi, mu, phi = net(x)
    assert pi.shape == (4, data.NUM_CATEGORIES)
    net, _, hist = train_federated(regions, ids, rounds=2, verbose=False)
    res = evaluate_federated(net, regions, ids)
    assert 0.0 <= res["global"]["macro_f1"] <= 1.0
    print("[ok] model forward + 2-round federated train, "
          f"global macro-F1={res['global']['macro_f1']*100:.1f}")


def test_all_strategies_run():
    if not HAS_TORCH:
        print("[skip] torch not installed -> strategy tests skipped")
        return
    from fedcrime.strategies import STRATEGIES, get_strategy
    from fedcrime.federated import train_federated
    regions = data.build_regions(DATA)
    ids = data.build_subset(regions, "alpha")
    for name in ["fedavg", "fedavgm", "fedprox", "scaffold", "fedtrimavg",
                 "multikrum", "bulyan", "moon", "fblg"]:
        strat = get_strategy(name)
        net, _, _ = train_federated(regions, ids, rounds=2, verbose=False,
                                    strategy=strat)
        assert net is not None
    print(f"[ok] all {len(STRATEGIES)} registered strategies train for 2 rounds")


if __name__ == "__main__":
    test_data_pipeline()
    test_subset_sparsity_ordering()
    test_zinb_math()
    test_model_and_train()
    test_all_strategies_run()
    print("\nAll available tests passed.")
