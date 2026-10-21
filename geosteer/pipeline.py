"""The ordered per-well pipeline. One code path serves both prediction and out-of-fold evaluation.

  1. neighbour marker surface        -> independent TVT estimate + sparse filter observations
  2. particle-filter ensemble         -> typewell or own-heel reference; filter / genealogy
                                         smoother / marker-filter outputs, regularised and mixed
  3. gated blend with the marker surface and the EM-plane member
  4. whole-well shape decoder
  5. heel taper, parallel-neighbour correction, start continuity
Also: data loading, out-of-fold evaluation, the submission writer and the command line.
"""

import argparse
import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import (AGREE_BOOST, EM_KNOT_STRIDE, EMPLANE_B0, EMPLANE_BMAX, EMPLANE_DIST0,
                     MARKER_A0, MARKER_AMAX, MARKER_DIS0, MARKER_DIST0, MARKER_SCALE, MIX_FILTER,
                     MIX_MARKER, MIX_SMOOTHER, OWN_REF_GATE_MARGIN, SMOOTH_BETA)
from .particle_filter import seed_ensemble
from .stages import (build_index, emplane_trajectory, heel_taper, marker_observations,
                     marker_transfer, parallel_correction, shape_decode, shape_regularize,
                     start_continuity)

log = logging.getLogger(__name__)


def well_names(data_dir: str, split: str) -> list[str]:
    """Sorted well ids present in `data_dir/split`."""
    files = sorted(glob.glob(os.path.join(data_dir, split, "*__horizontal_well.csv")))
    return [os.path.basename(f).replace("__horizontal_well.csv", "") for f in files]


def load_well(data_dir: str, split: str, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (horizontal log, typewell) for one well."""
    h = pd.read_csv(os.path.join(data_dir, split, f"{name}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(data_dir, split, f"{name}__typewell.csv"))
    return h, tw


def heel_tvt(h: pd.DataFrame) -> float:
    """Last known TVT: the hand-off point where prediction starts."""
    known = h["TVT_input"].dropna()
    return float(known.iloc[-1]) if len(known) else float(np.nanmean(h["TVT_input"]))


def filter_well(h: pd.DataFrame, tw: pd.DataFrame, rows: np.ndarray, mk_tvt=None, mk_sigma=None):
    """Particle-filter prediction for `rows` (stages 1-2) and its across-seed std.

    Two ensembles run: one on the typewell, one on the well's own heel GR(TVT). The own reference
    is adopted only when it is both a better fit (per-row log-likelihood) and sharper (lower
    across-seed spread); otherwise the typewell wins. Filter, smoother and marker-filter outputs are
    regularised separately, then mixed: their errors are only partly correlated.
    """
    known = h["TVT_input"].notna().values
    t0 = heel_tvt(h)
    ev = np.where(~known)[0]
    if len(ev) == 0: return np.full(len(rows), t0), np.zeros(len(rows))
    MD = h["MD"].values.astype(float)
    Z = h["Z"].values.astype(float)
    typewell_run = seed_ensemble(h, tw, own_reference=False, mk_tvt=mk_tvt, mk_sigma=mk_sigma)
    own_run = seed_ensemble(h, tw, own_reference=True, mk_tvt=mk_tvt, mk_sigma=mk_sigma)
    use_own = (
        own_run.loglik_per_row > typewell_run.loglik_per_row + OWN_REF_GATE_MARGIN
        and own_run.spread < typewell_run.spread
    )
    chosen = own_run if use_own else typewell_run
    pos = np.clip(np.searchsorted(ev, np.asarray(rows)), 0, len(ev) - 1)

    def reg(drift):
        return shape_regularize(drift, t0, MD[ev], Z[ev])

    filt_path, smooth_path = reg(chosen.filtered), reg(chosen.smoothed)
    if chosen.marker is not None:
        final = (
            MIX_FILTER * filt_path + MIX_SMOOTHER * smooth_path + MIX_MARKER * reg(chosen.marker)
        )
    else:
        final = (1.0 - SMOOTH_BETA) * filt_path + SMOOTH_BETA * smooth_path
    return final[pos], chosen.row_std[pos]


def stack_members(pbase, marker_tv, marker_nd, e_rows):
    """Blend the filter prediction with the marker surface and the EM-plane member (stage 3).

    Marker weight falls with filter/marker disagreement and with neighbour distance; the EM-plane
    weight grows where neighbours are far (it needs none). When both members push the same way
    (positive correlation of their corrections) both weights get a boost.
    """
    agreement = 0.0
    if marker_tv is not None and e_rows is not None:
        finite = np.isfinite(marker_tv) & np.isfinite(e_rows)
        if finite.sum() >= 3:
            d_marker = marker_tv[finite] - pbase[finite]
            d_emplane = e_rows[finite] - pbase[finite]
            if d_marker.std() > 0 and d_emplane.std() > 0:
                agreement = float(np.corrcoef(d_marker, d_emplane)[0, 1])
                if not np.isfinite(agreement): agreement = 0.0
    boost = 1.0 + AGREE_BOOST * max(agreement, 0.0)
    blended = pbase.copy()
    used_marker = False
    if marker_tv is not None and marker_nd is not None:
        finite = np.isfinite(marker_tv)
        if finite.any():
            dis = float(np.sqrt(np.mean((marker_tv[finite] - pbase[finite]) ** 2)))
            amax = min(
                min(MARKER_A0 * np.exp(-dis / MARKER_DIS0), MARKER_AMAX) * boost, MARKER_AMAX
            )
            a = np.where(finite, amax * np.exp(-marker_nd / MARKER_DIST0), 0.0) * MARKER_SCALE
            a = np.where(a > 1e-3, a, 0.0)
            blended = (1 - a) * blended + a * np.where(finite, marker_tv, 0.0)
            used_marker = True
    if e_rows is not None:
        em_finite = np.isfinite(e_rows)
        b0 = min(EMPLANE_B0 * boost, EMPLANE_BMAX)
        if marker_tv is not None and marker_nd is not None:
            b = np.where(em_finite, b0 * (1.0 - np.exp(-marker_nd / EMPLANE_DIST0)), 0.0)
        else:
            b = np.where(em_finite, b0, 0.0)
        blended = (1 - b) * blended + b * np.where(em_finite, e_rows, 0.0)
    return blended, used_marker


def _emplane_rows(h, tw, rows):
    ev, p = emplane_trajectory(h, tw)
    if ev is None: return None
    knots = np.arange(0, len(ev), EM_KNOT_STRIDE)
    full = np.interp(np.arange(len(ev)), knots, p[knots])
    pm = np.searchsorted(ev, rows)
    if pm.max() < len(ev) and np.all(ev[pm] == rows): return full[pm]
    return None


def predict_well(name: str, rows: np.ndarray, data_dir: str, split: str, index: dict):
    """Full pipeline for one well.

    Each stage falls back to "skip" if it raises, so one bad well cannot break a run; the number of
    such fallbacks is returned so callers can refuse the result (see `run_pipeline(strict=True)`).
    Returns (TVT per row, across-seed std per row, dict of which stages fired, fallback count).
    """
    rows = np.asarray(rows)
    fired = dict(
        marker=False, emplane=False, decoder=False, taper=False, parallel=False, continuity=False
    )
    errors = 0

    def failed(stage, e):
        nonlocal errors
        errors += 1
        log.warning("%s failed on well %s: %s", stage, name, e)

    h, tw = load_well(data_dir, split, name)
    try:
        marker_tv, marker_nd = marker_transfer(h, tw, rows, index, name)
    except Exception as e:
        failed("marker transfer", e)
        marker_tv, marker_nd = None, None
    try:
        pred, std = filter_well(h, tw, rows, *marker_observations(marker_tv, marker_nd))
    except Exception as e:  # last-resort fallback: flat at the heel TVT
        failed("filter", e)
        kv = h["TVT_input"].dropna()
        pred = np.full(len(rows), float(kv.iloc[-1]) if len(kv) else 0.0)
        std = np.full(len(rows), np.nan)

    try:
        e_rows = _emplane_rows(h, tw, rows)
    except Exception as e:
        failed("EM-plane", e)
        e_rows = None
    fired["emplane"] = e_rows is not None
    pred, fired["marker"] = stack_members(pred, marker_tv, marker_nd, e_rows)

    try:
        pc = shape_decode(h, tw, rows, pred)
        if pc is not pred and np.all(np.isfinite(pc)): pred, fired["decoder"] = pc, True
    except Exception as e:
        failed("shape decoder", e)

    steps = [
        ("taper", lambda p: heel_taper(h, rows, p)),
        ("parallel", lambda p: parallel_correction(h, tw, rows, p, index, name)),
        ("continuity", lambda p: start_continuity(h, p)),
    ]
    for stage, fn in steps:
        try:
            out = fn(pred)
        except Exception as e:
            failed(stage, e)
            out = None
        if out is not None: pred, fired[stage] = out, True
    return pred, std, fired, errors


_CTX: dict = {}


def _init_worker(ctx: dict) -> None:
    _CTX.update(ctx)


def _work(task):
    name, rows = task
    return name, *predict_well(name, rows, **_CTX)


def run_pipeline(
    data_dir: str,
    split: str,
    wells: list[tuple[str, np.ndarray]],
    workers: int | None = None,
    strict: bool = True,
):
    """Predict TVT for every (well, rows) task.

    Returns ({well: TVT per row}, {well: across-seed std per row}). With strict=True (default) the
    run fails if any stage fell back on any well, so a score or a submission always describes
    the full pipeline. workers=1 runs serially in this process.
    """
    ctx = dict(data_dir=data_dir, split=split, index=build_index(data_dir, strict=strict))
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        _init_worker(ctx)
        results = map(_work, wells)
    else:
        pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(ctx,))
        results = pool.map(_work, wells)
    preds, stds, counts, errors = {}, {}, {}, 0
    try:
        for name, pred, std, fired, err in results:
            preds[name] = pred
            stds[name] = std
            errors += err
            for k, v in fired.items(): counts[k] = counts.get(k, 0) + int(v)
    finally:
        if workers > 1: pool.shutdown()
    log.info(
        "stages fired (wells): %s; stage fallbacks: %d",
        ", ".join(f"{k} {v}/{len(wells)}" for k, v in counts.items()),
        errors,
    )
    if strict and errors:
        raise RuntimeError(f"{errors} stage fallbacks (see log); rerun with strict=False to accept")
    bad = [n for n, p in preds.items() if not np.all(np.isfinite(p))]
    if len(preds) != len(wells) or bad:
        raise RuntimeError(f"missing or non-finite predictions for wells: {bad[:5]}")
    for stage in ("marker", "emplane", "decoder"):
        if counts.get(stage, 0) < 0.5 * len(wells):
            log.warning("%s fired on fewer than half of the wells; check the input data", stage)
    return preds, stds


# --- out-of-fold evaluation -----------------------------------------------------------------------
# Each training well is treated exactly like a test well: only its known heel is visible and its
# own points are excluded from every neighbour cloud.


def train_tasks(data_dir: str, n_wells: int | None = None):
    """(tasks, truth): the hidden rows of the first `n_wells` training wells and their true TVT."""
    tasks, truth = [], {}
    for name in well_names(data_dir, "train")[:n_wells]:
        h = pd.read_csv(
            os.path.join(data_dir, "train", f"{name}__horizontal_well.csv"),
            usecols=["TVT", "TVT_input"],
        )
        ev = np.where(h["TVT_input"].isna().values)[0]
        if len(ev):
            tasks.append((name, ev))
            truth[name] = h["TVT"].values[ev].astype(float)
    return tasks, truth


def evaluate(data_dir: str, n_wells: int | None = None, workers: int | None = None):
    """Run the pipeline out-of-fold; return (predictions, std, truth, summary dict)."""
    tasks, truth = train_tasks(data_dir, n_wells)
    preds, stds = run_pipeline(data_dir, "train", tasks, workers=workers, strict=True)
    names = [n for n, _ in tasks]
    err = np.concatenate([preds[n] - truth[n] for n in names])
    per_well = np.array([np.sqrt(np.mean((preds[n] - truth[n]) ** 2)) for n in names])
    summary = dict(
        wells=len(tasks),
        rows=len(err),
        pooled_rmse=float(np.sqrt(np.mean(err**2))),
        median_well_rmse=float(np.median(per_well)),
        frac_wells_over_10ft=float(np.mean(per_well > 10.0)),
    )
    return preds, stds, truth, summary


# --- submission -----------------------------------------------------------------------------------


def write_submission(
    data_dir: str, out_path: str, workers: int | None = None, strict: bool = True
) -> pd.DataFrame:
    """Write `out_path` with columns id, tvt (id = "{well}_{row}") and return it."""
    path = os.path.join(data_dir, "sample_submission.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; --data must point at the competition data")
    sub = pd.read_csv(path)
    sub["well"] = sub["id"].str.rsplit("_", n=1).str[0]
    sub["row"] = sub["id"].str.rsplit("_", n=1).str[1].astype(int)
    test_wells = set(well_names(data_dir, "test"))
    groups = list(sub.groupby("well"))
    missing = [w for w, _ in groups if w not in test_wells]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} wells of sample_submission.csv have no files "
            f"under {os.path.join(data_dir, 'test')}, e.g. {missing[:3]}"
        )
    preds, _ = run_pipeline(
        data_dir, "test", [(w, g["row"].values) for w, g in groups], workers=workers, strict=strict
    )
    tvt = {i: v for w, g in groups for i, v in zip(g["id"].values, preds[w], strict=True)}
    sub["tvt"] = sub["id"].map(tvt)
    n_missing = int(sub["tvt"].isna().sum())
    if n_missing: raise RuntimeError(f"{n_missing} submission rows have no prediction")
    sub[["id", "tvt"]].to_csv(out_path, index=False)
    log.info("wrote %s: %d rows", out_path, len(sub))
    return sub


# --- command line: geosteer-predict, geosteer-evaluate --------------------------------------------


def _setup(ap: argparse.ArgumentParser) -> argparse.Namespace:
    ap.add_argument(
        "--data",
        required=True,
        help="data dir with train/ (and test/, " "sample_submission.csv for prediction)",
    )
    ap.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return args


def predict_main() -> None:
    """Write a submission for the wells in sample_submission.csv."""
    ap = argparse.ArgumentParser(description=predict_main.__doc__)
    ap.add_argument("--out", default="submission.csv", help="output CSV (default: submission.csv)")
    ap.add_argument(
        "--allow-fallback",
        action="store_true",
        help="write the file even if a stage fell back on some well",
    )
    args = _setup(ap)
    write_submission(args.data, args.out, args.workers, strict=not args.allow_fallback)


def evaluate_main() -> None:
    """Self-excluded out-of-fold RMSE on the first N training wells."""
    ap = argparse.ArgumentParser(description=evaluate_main.__doc__)
    ap.add_argument("--wells", type=int, default=None, help="first N wells by id (default: all)")
    ap.add_argument("--save-preds", default=None, help="write per-well predictions to this .npz")
    args = _setup(ap)
    preds, _, _, s = evaluate(args.data, args.wells, args.workers)
    if args.save_preds: np.savez_compressed(args.save_preds, **preds)
    print(f"wells {s['wells']}  rows {s['rows']}")
    print(f"pooled RMSE          {s['pooled_rmse']:.4f} ft")
    print(f"median well RMSE     {s['median_well_rmse']:.4f} ft")
    print(f"wells over 10 ft     {100 * s['frac_wells_over_10ft']:.1f}%")


if __name__ == "__main__":
    import sys

    {"predict": predict_main, "evaluate": evaluate_main}[sys.argv.pop(1)]()
