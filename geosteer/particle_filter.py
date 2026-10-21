"""Sequential Monte Carlo over the well's stratigraphic position.

Each particle carries the structural datum u = TVT + Z and its rate du/dMD. Geology is smooth in
u; the fast wiggles of TVT come from the surveyed trajectory Z, and TVT = u - Z.

Two emissions score a particle against the measured gamma ray, with G the reference GR(TVT):
  * level: GR ~ G(TVT);
  * flank: dGR/dMD ~ G'(TVT) * (rate - dZ/dMD), by the chain rule. For a given direction of
    travel, two depths with the same GR level on opposite flanks of a bed predict opposite
    gradients, so this term separates them.

N_SEEDS independent filters (an island particle filter) are averaged with softmax(log-lik / T)
weights, which gives the right cycle several chances.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from .config import (CHI2_CUTOFF, DIP_PERSISTENCE, FLANK_WEIGHT, GR_SIGMA, GRID_STEP, HEEL_DIP_ROWS,
                     INIT_POS_SD, INIT_RATE_SD, LIK_FLOOR, LOGLIK_FLOOR, MARKER_TAIL_LIK,
                     N_PARTICLES, N_SEEDS, OWN_REF_MIN_COVER, OWN_REF_MIN_ROWS, POS_NOISE,
                     RATE_MOMENTUM, RATE_NOISE, RESAMPLE_ESS, RESAMPLE_POS_JITTER,
                     RESAMPLE_RATE_JITTER, SEED_TEMPERATURE, TVT_MARGIN)

# fast-math without the no-NaN / no-Inf assumptions: the kernels test for NaN GR explicitly
FASTMATH = {"reassoc", "contract", "arcp", "afn", "nsz"}


@njit(cache=True, fastmath=FASTMATH)
def _lookup(grid_vals, v, vmin, step):
    """Linear interpolation on a regular grid, clamped at both ends."""
    i = int((v - vmin) / step)
    n = len(grid_vals) - 1
    if i < 0: return grid_vals[0]
    if i >= n: return grid_vals[n]
    t = (v - vmin) / step - i
    return grid_vals[i] * (1.0 - t) + grid_vals[i + 1] * t


@njit(cache=True, fastmath=FASTMATH)
def particle_smoother(md, z, gr, ref, vmin, gstep, gr_sigma, start_datum, start_rate, n_part, seed,
                      dgr, dref, dgr_sigma, flank_weight, persistence):
    """Filter with level + flank emissions, plus a genealogy smoother.

    Returns (filtered TVT, smoothed TVT, log-likelihood). The smoother follows each final particle's
    ancestry back through the resampling steps, so row i also benefits from GR measured after i
    (legal: GR is given for every row, only TVT is hidden).
    """
    np.random.seed(seed)
    n = len(md)
    pos = np.empty(n_part)
    rate = np.empty(n_part)
    w = np.empty(n_part)
    tvt_hist = np.empty((n, n_part))
    parent = np.empty((n, n_part), dtype=np.int32)
    for j in range(n_part):
        pos[j] = start_datum + INIT_POS_SD * np.random.randn()
        rate[j] = start_rate + INIT_RATE_SD * np.random.randn()
        w[j] = 1.0 / n_part
    filt = np.empty(n)
    log_lik = 0.0
    prev_md = md[0] - 1.0
    prev_z = z[0]
    nref = len(ref)
    # the dip rate mean-reverts toward a fraction of the heel dip (structure persists, but fades)
    pull = (1.0 - RATE_MOMENTUM) * persistence * start_rate
    use_flank = flank_weight > 0.0 and len(dgr) == n and len(dref) > 0
    for i in range(n):
        dm = md[i] - prev_md
        if dm < 1.0: dm = 1.0
        dzdm = (z[i] - prev_z) / dm
        for j in range(n_part):
            rate[j] = RATE_MOMENTUM * rate[j] + pull + RATE_NOISE * np.random.randn()
            pos[j] += rate[j] * dm + POS_NOISE * np.random.randn()
            tvt = pos[j] - z[i]
            if tvt < vmin - TVT_MARGIN: tvt = vmin - TVT_MARGIN
            if tvt > vmin + nref * gstep + TVT_MARGIN: tvt = vmin + nref * gstep + TVT_MARGIN
            pos[j] = tvt + z[i]
            tvt_hist[i, j] = pos[j] - z[i]
            parent[i, j] = j
        if not np.isnan(gr[i]):
            total_w = 0.0
            flank_ok = use_flank and not np.isnan(dgr[i]) and i > 0
            for j in range(n_part):
                expected_gr = _lookup(ref, pos[j] - z[i], vmin, gstep)
                d = (gr[i] - expected_gr) / gr_sigma
                z2 = d * d
                e = -0.5 * z2
                if flank_ok:
                    # expected dGR/dMD = G'(TVT) * (rate - dZ/dMD)
                    expected_dgr = _lookup(dref, pos[j] - z[i], vmin, gstep) * (rate[j] - dzdm)
                    df = (dgr[i] - expected_dgr) / dgr_sigma
                    e += -0.5 * flank_weight * df * df
                lik = np.exp(e) if e > LOGLIK_FLOOR else 0.0
                if lik < LIK_FLOOR: lik = LIK_FLOOR
                w[j] *= lik
                total_w += w[j]
            if total_w > 0:
                for j in range(n_part): w[j] /= total_w
                log_lik += np.log(total_w)
            else:
                for j in range(n_part): w[j] = 1.0 / n_part
                log_lik += np.log(LIK_FLOOR)
        sum_w2 = 0.0
        for j in range(n_part): sum_w2 += w[j] * w[j]
        mean_tvt = 0.0
        for j in range(n_part): mean_tvt += w[j] * (pos[j] - z[i])
        filt[i] = mean_tvt
        if 1.0 / sum_w2 < RESAMPLE_ESS * n_part:  # systematic resampling
            cum = np.empty(n_part)
            c = 0.0
            for j in range(n_part):
                c += w[j]
                cum[j] = c
            u_first = np.random.uniform(0.0, 1.0 / n_part)
            new_pos = np.empty(n_part)
            new_rate = np.empty(n_part)
            src = 0
            for j in range(n_part):
                u = u_first + j / n_part
                while src < n_part - 1 and cum[src] < u:
                    src += 1
                new_pos[j] = pos[src] + RESAMPLE_POS_JITTER * np.random.randn()
                new_rate[j] = rate[src] + RESAMPLE_RATE_JITTER * np.random.randn()
                parent[i, j] = src
            for j in range(n_part):
                pos[j] = new_pos[j]
                rate[j] = new_rate[j]
                w[j] = 1.0 / n_part
        prev_md = md[i]
        prev_z = z[i]
    smooth = np.zeros(n)
    lineage = np.empty(n_part, dtype=np.int32)
    for j in range(n_part): lineage[j] = j
    for i in range(n - 1, -1, -1):
        for j in range(n_part): smooth[i] += w[j] * tvt_hist[i, lineage[j]]
        if i > 0:
            for j in range(n_part): lineage[j] = parent[i, lineage[j]]
    return filt, smooth, log_lik


@njit(cache=True, fastmath=FASTMATH)
def marker_filter(
    md, z, gr, ref, vmin, gstep, gr_sigma, start_datum, start_rate, n_part, seed, mk_tvt, mk_sigma
):
    """Level-emission filter fused with sparse neighbour-marker observations.

    Rows with mk_sigma > 0 carry a Gaussian observation N(TVT; mk_tvt, mk_sigma).
    Returns (filtered TVT, log-likelihood). This arm scores the GR level only.
    """
    np.random.seed(seed)
    pos = np.empty(n_part)
    rate = np.empty(n_part)
    w = np.empty(n_part)
    for j in range(n_part):
        pos[j] = start_datum + INIT_POS_SD * np.random.randn()
        rate[j] = start_rate + INIT_RATE_SD * np.random.randn()
        w[j] = 1.0 / n_part
    n = len(md)
    out = np.empty(n)
    log_lik = 0.0
    prev_md = md[0] - 1.0
    nref = len(ref)
    pull = (1.0 - RATE_MOMENTUM) * DIP_PERSISTENCE * start_rate
    for i in range(n):
        dm = md[i] - prev_md
        if dm < 1.0: dm = 1.0
        for j in range(n_part):
            rate[j] = RATE_MOMENTUM * rate[j] + pull + RATE_NOISE * np.random.randn()
            pos[j] += rate[j] * dm + POS_NOISE * np.random.randn()
            tvt = pos[j] - z[i]
            if tvt < vmin - TVT_MARGIN: tvt = vmin - TVT_MARGIN
            if tvt > vmin + nref * gstep + TVT_MARGIN: tvt = vmin + nref * gstep + TVT_MARGIN
            pos[j] = tvt + z[i]
        has_gr = not np.isnan(gr[i])
        has_mk = mk_sigma[i] > 0.0
        if has_gr or has_mk:
            total_w = 0.0
            for j in range(n_part):
                lik = 1.0
                if has_gr:
                    expected_gr = _lookup(ref, pos[j] - z[i], vmin, gstep)
                    d = (gr[i] - expected_gr) / gr_sigma
                    z2 = d * d
                    lik *= np.exp(-0.5 * z2) if z2 < CHI2_CUTOFF else 0.0
                if has_mk:
                    z_marker = (pos[j] - z[i] - mk_tvt[i]) / mk_sigma[i]
                    z2_marker = z_marker * z_marker
                    lik *= np.exp(-0.5 * z2_marker) if z2_marker < CHI2_CUTOFF else MARKER_TAIL_LIK
                if lik < LIK_FLOOR: lik = LIK_FLOOR
                w[j] *= lik
                total_w += w[j]
            if total_w > 0:
                for j in range(n_part): w[j] /= total_w
                log_lik += np.log(total_w)
            else:
                for j in range(n_part): w[j] = 1.0 / n_part
                log_lik += np.log(LIK_FLOOR)
        sum_w2 = 0.0
        for j in range(n_part): sum_w2 += w[j] * w[j]
        if 1.0 / sum_w2 < RESAMPLE_ESS * n_part:
            cum = np.empty(n_part)
            c = 0.0
            for j in range(n_part):
                c += w[j]
                cum[j] = c
            u_first = np.random.uniform(0.0, 1.0 / n_part)
            new_pos = np.empty(n_part)
            new_rate = np.empty(n_part)
            src = 0
            for j in range(n_part):
                u = u_first + j / n_part
                while src < n_part - 1 and cum[src] < u:
                    src += 1
                new_pos[j] = pos[src] + RESAMPLE_POS_JITTER * np.random.randn()
                new_rate[j] = rate[src] + RESAMPLE_RATE_JITTER * np.random.randn()
            for j in range(n_part):
                pos[j] = new_pos[j]
                rate[j] = new_rate[j]
                w[j] = 1.0 / n_part
        mean_tvt = 0.0
        for j in range(n_part): mean_tvt += w[j] * (pos[j] - z[i])
        out[i] = mean_tvt
        prev_md = md[i]
    return out, log_lik


def typewell_curve(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Typewell GR as a function of TVT: finite samples, sorted by TVT."""
    tvt = tw["TVT"].values.astype(float)
    gr = tw["GR"].values.astype(float)
    ok = np.isfinite(tvt) & np.isfinite(gr)
    tvt, gr = tvt[ok], gr[ok]
    order = np.argsort(tvt)
    return tvt[order], gr[order]


def _own_reference(
    h: pd.DataFrame, gr: np.ndarray, grid: np.ndarray, twt: np.ndarray, twg: np.ndarray
):
    """The well's own GR(TVT) from its known heel, affinely mapped to the typewell scale.

    Same logging tool and higher resolution than the typewell, so its shape matches the lateral
    better. Returns (coverage mask on `grid`, own curve on `grid`) or None if the heel is too short.
    """
    tvi = h["TVT_input"].values.astype(float)
    km = np.isfinite(tvi)
    if km.sum() < OWN_REF_MIN_ROWS: return None
    ot = tvi[km]
    og = gr[km]
    o = np.argsort(ot)
    ot, og = ot[o], og[o]
    cover = (grid >= ot.min()) & (grid <= ot.max())
    if cover.sum() <= OWN_REF_MIN_COVER: return None
    tg = np.interp(grid[cover], twt, twg)
    ov = np.interp(grid[cover], ot, og)
    if not ov.std() > 1e-6: return None
    a, b = np.polyfit(ov, tg, 1)
    return cover, a * np.interp(grid, ot, og) + b


def _seed_weights(loglik: np.ndarray) -> np.ndarray:
    w = np.exp((loglik - loglik.max()) / SEED_TEMPERATURE)
    return w / w.sum()


@dataclass
class FilterInputs:
    """Everything the kernels need for one well's hidden rows."""

    ev: np.ndarray  # hidden row indices
    md: np.ndarray  # MD, Z and gap-interpolated GR on those rows
    z: np.ndarray
    gr: np.ndarray
    ref: np.ndarray  # reference GR on the regular TVT grid starting at vmin
    vmin: float
    start_datum: float  # u = TVT + Z at the last known row
    heel_rate: float  # median du/dMD over the last HEEL_DIP_ROWS known rows
    last_tvt: float
    dgr: np.ndarray  # measured dGR/dMD (empty if the flank term is off)
    dref: np.ndarray  # reference dG/dTVT on the grid
    dgr_sigma: float


def filter_inputs(h: pd.DataFrame, tw: pd.DataFrame, own_reference: bool) -> FilterInputs | None:
    """Reference curves, heel state and flank inputs; None if nothing is hidden or no typewell.

    GR gaps are filled by linear interpolation before scoring, so every row has a GR value.
    """
    known = h["TVT_input"].notna().values
    ev = np.where(~known)[0]
    twt, twg = typewell_curve(tw)
    if len(ev) == 0 or len(twt) < 5: return None
    vmin = twt[0]
    grid = np.arange(vmin, twt[-1] + GRID_STEP, GRID_STEP)
    ref = np.interp(grid, twt, twg)
    MD = h["MD"].values.astype(float)
    Z = h["Z"].values.astype(float)
    gr = h["GR"].interpolate(limit_direction="both").fillna(twg.mean()).values.astype(float)
    own = _own_reference(h, gr, grid, twt, twg)
    if own_reference and own is not None: ref = np.where(own[0], own[1], ref)

    kn = h[known]
    last_tvt = float(kn["TVT_input"].values[-1])
    last_z = float(kn["Z"].values[-1])
    tail = kn.tail(HEEL_DIP_ROWS)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dmd = np.diff(tail["MD"].values)
    mm = dmd > 0
    heel_rate = float(np.median((dt + dz)[mm] / dmd[mm])) if mm.sum() >= 3 else 0.0
    md_ev = MD[ev]
    gr_ev = gr[ev]

    # flank emission inputs: measured dGR/dMD and the reference gradient dG/dTVT. The gradient
    # reference is the well's own heel GR where available (same tool: gradient shapes match).
    dgr = np.zeros(0)
    dref = np.zeros(0)
    dgr_sigma = 1.0
    if FLANK_WEIGHT > 0.0 and len(md_ev) > 2:
        dgr = np.gradient(gr_ev, md_ev + 1e-9)
        grad_ref = np.where(own[0], own[1], ref) if own is not None else ref
        dref = np.gradient(grad_ref, GRID_STEP)
        dgr_sigma = float(np.nanstd(dgr)) + 1e-6
    return FilterInputs(
        ev,
        md_ev,
        Z[ev],
        gr_ev,
        ref,
        vmin,
        last_tvt + last_z,
        heel_rate,
        last_tvt,
        dgr,
        dref,
        dgr_sigma,
    )


@dataclass
class FilterOutput:
    filtered: np.ndarray  # TVT drift from the heel, per eval row
    smoothed: np.ndarray
    marker: np.ndarray | None  # marker-observed filter drift (None if no marker observation)
    spread: float  # median across-seed posterior std (lower = sharper)
    row_std: np.ndarray  # across-seed std of the smoothed TVT, per eval row
    loglik_per_row: float  # best run's log-likelihood per row (reference quality)


def seed_ensemble(
    h: pd.DataFrame,
    tw: pd.DataFrame,
    own_reference: bool,
    mk_tvt: np.ndarray | None = None,
    mk_sigma: np.ndarray | None = None,
) -> FilterOutput:
    """Island particle filter on the unknown part of one well.

    N_SEEDS independent filters run on the same data; each is weighted by softmax of its marginal
    log-likelihood divided by SEED_TEMPERATURE (T = 2).
    """
    fi = filter_inputs(h, tw, own_reference)
    if fi is None:
        zeros = np.zeros(int((~h["TVT_input"].notna().values).sum()))
        return FilterOutput(zeros, zeros, None, 0.0, zeros, -1e30)
    n = len(fi.ev)
    preds = np.empty((N_SEEDS, n))
    smooths = np.empty((N_SEEDS, n))
    lls = np.empty(N_SEEDS)
    for s in range(N_SEEDS):
        preds[s], smooths[s], lls[s] = particle_smoother(fi.md, fi.z, fi.gr, fi.ref, fi.vmin,
                                                         GRID_STEP, GR_SIGMA, fi.start_datum,
                                                         fi.heel_rate, N_PARTICLES, s, fi.dgr,
                                                         fi.dref, fi.dgr_sigma, FLANK_WEIGHT,
                                                         DIP_PERSISTENCE)
    wts = _seed_weights(lls)
    mu_f = (wts[:, None] * preds).sum(0)
    mu_s = (wts[:, None] * smooths).sum(0)
    spread = float(np.median(np.sqrt((wts[:, None] * (preds - mu_f) ** 2).sum(0))))
    row_std = np.sqrt((wts[:, None] * (smooths - mu_s) ** 2).sum(0))

    marker = None
    if mk_tvt is not None and mk_sigma is not None and np.any(np.asarray(mk_sigma) > 0):
        mtv = np.ascontiguousarray(np.asarray(mk_tvt, float))
        ms = np.ascontiguousarray(np.asarray(mk_sigma, float))
        preds_m = np.empty((N_SEEDS, n))
        lls_m = np.empty(N_SEEDS)
        for s in range(N_SEEDS):
            preds_m[s], lls_m[s] = marker_filter(fi.md, fi.z, fi.gr, fi.ref, fi.vmin, GRID_STEP,
                                                 GR_SIGMA, fi.start_datum, fi.heel_rate,
                                                 N_PARTICLES, s, mtv, ms)
        marker = (_seed_weights(lls_m)[:, None] * preds_m).sum(0) - fi.last_tvt
    return FilterOutput(
        mu_f - fi.last_tvt,
        mu_s - fi.last_tvt,
        marker,
        spread,
        row_std,
        float(lls.max()) / max(1, n),
    )
