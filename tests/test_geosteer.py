"""Data-free tests on synthetic wells.

A synthetic well: a typewell whose GR repeats every 8.4 ft (plus a slower component), a datum
u = TVT + Z with a constant dip, and a trajectory that porpoises a few feet, so it crosses GR
cycles in both directions. Measured GR = typewell(TVT) + smooth noise. Under a minute in total.
"""

import os

import numpy as np
import pandas as pd
import pytest

from geosteer.analysis import TABLE, summary_numbers
from geosteer.particle_filter import particle_smoother
from geosteer.pipeline import evaluate, filter_well, write_submission

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STEP, DIP, TVT0 = 0.5, 0.004, 11000.0


def synthetic_well(n=2500, heel=800, noise=3.0, x0=1000.0, seed=0):
    """(horizontal log with TVT, typewell, true TVT)."""
    t = np.arange(TVT0, TVT0 + 300.0, STEP)
    g = 80 + 30 * np.sin(2 * np.pi * t / 8.4) + 12 * np.sin(2 * np.pi * t / 23 + 1.0)
    md = 12000.0 + np.arange(n, dtype=float)
    z = -9000 - 4.0 * np.sin(2 * np.pi * md / 900 + 1.0)
    tvt = TVT0 + 150 - 9000 + DIP * (md - md[0]) - z
    noise_ = np.convolve(np.random.default_rng(seed).standard_normal(n + 8), np.ones(9) / 3, "valid")
    h = pd.DataFrame(dict(MD=md, X=x0 + md - md[0], Y=5000 + 0.1 * (md - md[0]), Z=z,
                          GR=np.interp(tvt, t, g) + noise * noise_[:n], TVT=tvt,
                          TVT_input=np.where(np.arange(n) < heel, tvt, np.nan)))
    return h, pd.DataFrame(dict(TVT=t, GR=g)), tvt


def test_flank_chain_rule():
    """Noise-free: dGR/dMD equals G'(TVT) * (rate - dZ/dMD), the identity behind the flank term."""
    h, tw, tvt = synthetic_well(noise=0.0)
    md = h["MD"].values
    observed = np.gradient(h["GR"].values, md)
    gradient = np.interp(tvt, tw["TVT"], np.gradient(tw["GR"].values, STEP))
    expected = gradient * (DIP - np.gradient(h["Z"].values, md))
    assert np.corrcoef(observed[5:-5], expected[5:-5])[0, 1] > 0.99


def test_filter_is_deterministic():
    h, tw, tvt = synthetic_well()
    ev = np.where(h["TVT_input"].isna())[0]
    md, z, gr = h["MD"].values[ev], h["Z"].values[ev], h["GR"].values[ev]
    dgr, dref = np.gradient(gr, md), np.gradient(tw["GR"].values, STEP)
    args = (md, z, gr, tw["GR"].values, TVT0, STEP, 45.0, tvt[ev[0] - 1] + z[0], DIP, 300, 7,
            dgr, dref, float(np.std(dgr)), 1.0, 0.6)
    a, b = particle_smoother(*args), particle_smoother(*args)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_filter_recovers_tvt():
    """The full filter stage (128 runs, both references, mixture) tracks TVT to within 0.5 ft."""
    h, tw, tvt = synthetic_well()
    rows = np.where(h["TVT_input"].isna())[0]
    pred, _ = filter_well(h.drop(columns="TVT"), tw, rows)
    assert np.sqrt(np.mean((pred - tvt[rows]) ** 2)) < 0.5
    assert abs(pred[0] - tvt[rows[0] - 1]) < 2.0      # continuous at the hand-off


def test_end_to_end(tmp_path):
    """evaluate + write_submission on 4 training wells and 1 test well, serially."""
    data = tmp_path / "data"
    for split in ("train", "test"):
        os.makedirs(data / split)
    for i, x0 in enumerate([1000, 1150, 1300, 1450, 1075]):
        h, tw, _ = synthetic_well(n=320, heel=120, noise=2.0, x0=x0, seed=i)
        split, name = ("train", f"w{i}") if i < 4 else ("test", "t0")
        (h if i < 4 else h.drop(columns="TVT")).to_csv(
            data / split / f"{name}__horizontal_well.csv", index=False)
        tw.to_csv(data / split / f"{name}__typewell.csv", index=False)
    rows = np.where(h["TVT_input"].isna())[0]
    pd.DataFrame(dict(id=[f"t0_{r}" for r in rows], tvt=0.0)).to_csv(
        data / "sample_submission.csv", index=False)
    _, _, _, s = evaluate(str(data), workers=1)
    assert s["wells"] == 4 and s["pooled_rmse"] < 2.0
    sub = write_submission(str(data), str(tmp_path / "submission.csv"), workers=1)
    assert len(sub) == 200 and np.all(np.isfinite(sub["tvt"]))


def test_bad_data_dir_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate(str(tmp_path), workers=1)


def test_results_table_matches_readme():
    """The README's error-analysis numbers are what geosteer.analysis computes."""
    s = summary_numbers(pd.read_csv(TABLE))
    readme = open(os.path.join(ROOT, "README.md")).read()
    for q in [f"{s['pooled']:.3f} ft", f"{s['median_well']:.2f} ft",
              f"{s['over10_n']} ({100 * s['over10_frac']:.1f}%)",
              f"{100 * s['over10_sse_share']:.1f}%",
              f"{s['per_tenth'][0]:.2f} → {s['per_tenth'][-1]:.2f} ft",
              f"{100 * (s['offset_share'] + s['dip_share']):.1f}%",
              "{:.1f}% / {:.1f}% / {:.1f}%".format(
                  100 * s["offset_share"], 100 * s["dip_share"], 100 * s["shape_share"])]:
        assert q in readme, q
