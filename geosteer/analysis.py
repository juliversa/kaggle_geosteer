"""Error analysis of the out-of-fold predictions, from the per-well table in results/.

    python -m geosteer.analysis                          # summary from results/oof_per_well.csv
    python -m geosteer.analysis --plot growth.png        # also draw pooled RMSE per tenth
    python -m geosteer.analysis --build oof.npz --data DATA          # rebuild the table
    python -m geosteer.analysis --model-score oof.npz --data DATA    # full-model comparison

`oof.npz` holds one array per well (predicted TVT on its hidden rows, in row order), as written by
`geosteer-evaluate --save-preds`. Everything except the summary needs the competition data,
which is not redistributed; the table holds only per-well aggregates.

Per-well error decomposition, additive (residual r = prediction - truth over the hidden rows):
  offset = n * mean(r)^2                          (a constant vertical shift)
  dip    = squared error removed by a straight line in MD fitted to r - mean(r)
  shape  = what remains;   offset + dip + shape = sum(r^2), so the shares sum to 100%.
"""

import argparse
import os

import numpy as np
import pandas as pd

TABLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "oof_per_well.csv"
)
TENTHS = [f"rmse_t{k}" for k in range(10)]


def tenth_slices(n: int):
    """Row ranges of the 10 consecutive tenths of a blind stretch of n rows (heel to toe)."""
    edges = np.linspace(0, n, 11).astype(int)
    return [slice(edges[k], edges[k + 1]) for k in range(10)]


def decompose(md: np.ndarray, r: np.ndarray) -> dict:
    """Additive offset / dip / shape split of the squared error of one well."""
    n = len(r)
    sse = float((r**2).sum())
    r1 = r - r.mean()
    t = (md - md.min()) / (np.ptp(md) + 1e-9)
    coef = np.linalg.lstsq(np.vstack([t, np.ones(n)]).T, r1, rcond=None)[0]
    r2 = r1 - (coef[0] * t + coef[1])
    return dict(
        offset_share=n * r.mean() ** 2 / sse,
        dip_share=float((r1**2).sum() - (r2**2).sum()) / sse,
        shape_share=float((r2**2).sum()) / sse,
    )


def _train(data: str, name: str):
    h = pd.read_csv(os.path.join(data, "train", f"{name}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(data, "train", f"{name}__typewell.csv"))
    return h, tw, np.where(h["TVT_input"].isna().values)[0]


def build(preds_path: str, data: str, out: str) -> None:
    preds = np.load(preds_path)
    names = sorted(preds.files)
    rows = []
    for w in names:
        h, _, ev = _train(data, w)
        r = preds[w] - h["TVT"].values[ev]
        row = dict(well=w, rows=len(r), rmse=float(np.sqrt(np.mean(r**2))),
                   **decompose(h["MD"].values[ev], r))
        for k, sl in enumerate(tenth_slices(len(r))):
            row[TENTHS[k]] = float(np.sqrt(np.mean(r[sl] ** 2)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}: {len(rows)} wells")


def bad_wells(preds, data, min_rmse=10.0):
    for name in sorted(preds.files):
        h, tw, ev = _train(data, name)
        p, t = preds[name], h["TVT"].values[ev]
        if np.sqrt(np.mean((p - t) ** 2)) > min_rmse: yield name, h, tw, ev, p, t


def _lookup(vals, v, vmin, step):
    """Vectorised version of the filter's clamped grid interpolation."""
    x = (v - vmin) / step
    i = np.trunc(x).astype(int)  # the kernel uses int(), which truncates
    hi = len(vals) - 1
    t = x - i
    ic = np.clip(i, 0, hi - 1)
    out = vals[ic] * (1.0 - t) + vals[ic + 1] * t
    return np.where(i < 0, vals[0], np.where(i >= hi, vals[hi], out))


def path_log_joint(fi, tvt: np.ndarray) -> dict:
    """log p(path, GR) of a TVT path under the filter's own model, split by term.

    The path fixes u = TVT + Z and its rate du/dMD (row differences). Terms:
      level     sum of per-row Gaussian GR log-likelihoods (GR_SIGMA), with the filter's floor
      emission  level plus the flank term inside the same floor, as in the kernel
      prior     AR(1) rate prior with the datum jitter POS_NOISE: the exact linear-Gaussian
                marginal of the path's increments, the latent rate integrated out
    Resampling jitter is part of the algorithm, not of the model, and is not scored.
    """
    from .config import (DIP_PERSISTENCE, FLANK_WEIGHT, GR_SIGMA, GRID_STEP, INIT_POS_SD,
                                 INIT_RATE_SD, LIK_FLOOR, LOGLIK_FLOOR, POS_NOISE, RATE_MOMENTUM,
                                 RATE_NOISE)

    md, z, gr = fi.md, fi.z, fi.gr
    n = len(md)
    u = tvt + z
    dm = np.maximum(np.diff(np.r_[md[0] - 1.0, md]), 1.0)
    du = np.diff(np.r_[fi.start_datum, u])
    rate = du / dm
    dzdm = np.r_[0.0, np.diff(z)] / dm
    e = -0.5 * ((gr - _lookup(fi.ref, tvt, fi.vmin, GRID_STEP)) / GR_SIGMA) ** 2
    level = np.where(e > LOGLIK_FLOOR, e, np.log(LIK_FLOOR)).sum()
    if len(fi.dgr) == n:
        edg = _lookup(fi.dref, tvt, fi.vmin, GRID_STEP) * (rate - dzdm)
        ef = -0.5 * FLANK_WEIGHT * ((fi.dgr - edg) / fi.dgr_sigma) ** 2
        ef[0] = 0.0
        ef[~np.isfinite(fi.dgr)] = 0.0
        e = e + ef
    emission = np.where(e > LOGLIK_FLOOR, e, np.log(LIK_FLOOR)).sum()
    pull = (1.0 - RATE_MOMENTUM) * DIP_PERSISTENCE * fi.heel_rate
    m, P, kal = fi.heel_rate, INIT_RATE_SD**2, 0.0
    for i in range(n):
        m = RATE_MOMENTUM * m + pull
        P = RATE_MOMENTUM**2 * P + RATE_NOISE**2
        R = POS_NOISE**2 + (INIT_POS_SD**2 if i == 0 else 0.0)
        S = dm[i] ** 2 * P + R
        res = du[i] - dm[i] * m
        kal += -0.5 * (res * res / S + np.log(2 * np.pi * S))
        K = P * dm[i] / S
        m += K * res
        P *= 1.0 - K * dm[i]
    return dict(level=float(level), emission=float(emission), prior=kal)


def model_score(preds_path: str, data: str) -> None:
    """Which path does the filter's own model prefer, the truth or the prediction?"""
    from .particle_filter import filter_inputs

    keys = ["level", "emission", "prior", "joint"]
    wins = dict.fromkeys(keys, 0)
    n = 0
    for _, h, tw, _, p, t in bad_wells(np.load(preds_path), data):
        fi = filter_inputs(h, tw, own_reference=False)
        a, b = path_log_joint(fi, t), path_log_joint(fi, p)
        for s in (a, b): s["joint"] = s["emission"] + s["prior"]
        n += 1
        for k in keys: wins[k] += int(a[k] > b[k])
    print(f"wells over 10 ft: {n}; the true path scores higher than the prediction on:")
    for k in keys: print(f"  {k:13s} {wins[k]:3d} / {n}")


def summary_numbers(df: pd.DataFrame) -> dict:
    sse = df["rmse"] ** 2 * df["rows"]
    tot = sse.sum()
    over = df["rmse"] > 10
    counts = np.array([[sl.stop - sl.start for sl in tenth_slices(n)] for n in df["rows"]])
    per_tenth = np.sqrt((df[TENTHS].values ** 2 * counts).sum(0) / counts.sum(0))
    out = dict(
        wells=len(df),
        rows=int(df["rows"].sum()),
        pooled=float(np.sqrt(tot / df["rows"].sum())),
        median_well=float(df["rmse"].median()),
        over10_n=int(over.sum()),
        over10_frac=float(over.mean()),
        over10_sse_share=float(sse[over].sum() / tot),
        per_tenth=per_tenth.tolist(),
    )
    for part in ("offset", "dip", "shape"):
        out[part + "_share"] = float((df[part + "_share"] * sse).sum() / tot)
    return out


def summary(df: pd.DataFrame, plot: str | None = None) -> None:
    s = summary_numbers(df)
    print(f"wells {s['wells']}, rows {s['rows']}")
    print(f"pooled RMSE           {s['pooled']:.4f} ft")
    print(f"median well RMSE      {s['median_well']:.2f} ft")
    print(
        f"wells over 10 ft      {s['over10_n']} ({100 * s['over10_frac']:.1f}%), holding "
        f"{100 * s['over10_sse_share']:.1f}% of the squared error"
    )
    print(
        "squared error: offset {:.1f}%, dip {:.1f}%, shape {:.1f}%; offset + dip {:.1f}%".format(
            100 * s["offset_share"],
            100 * s["dip_share"],
            100 * s["shape_share"],
            100 * (s["offset_share"] + s["dip_share"]),
        )
    )
    print("pooled RMSE per tenth (heel -> toe): " + " ".join(f"{v:.2f}" for v in s["per_tenth"]))
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
        ax.bar(np.arange(1, 11), s["per_tenth"], color="#4c78a8")
        ax.set_xlabel("tenth of the blind stretch (1 = where the labels stop)")
        ax.set_ylabel("pooled RMSE (ft)")
        ax.set_xticks(np.arange(1, 11))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(plot)
        print("wrote", plot)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--plot", default=None, help="write the per-tenth error figure here")
    ap.add_argument("--build", default=None, metavar="OOF.npz", help="rebuild --table from these")
    ap.add_argument("--model-score", default=None, metavar="OOF.npz")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    if args.build: build(args.build, args.data, args.table)
    if args.model_score: model_score(args.model_score, args.data)
    summary(pd.read_csv(args.table), args.plot)


if __name__ == "__main__": main()
