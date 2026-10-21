"""The pipeline stages around the particle filter: neighbour-well marker surface, EM-plane member,
whole-well shape decoder, and the shape regulariser and final corrections."""

import glob
import logging
import math
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import (ANCHOR_K, ANISO_K, CLOUD_QUERY_K, CLOUD_STRIDE, CROSS_TRACK_SCALE,
                     DENSE_STRIDE, EM_BAND, EM_DECIM, EM_GR_SIGMA, EM_HEEL_ROWS, EM_ITERS,
                     EM_K_RANGE, EM_RIDGE, EM_START_SIGMA, EM_STATE_STEP, EM_TRANS_SIGMA, EM_WINDOW,
                     HEEL_MAX_DIP, HEEL_TAPER_ROWS, IRLS_ITERS, MARKER_QUERY_STRIDE, MIN_CLOUD_ROWS,
                     OBS_EVERY, OBS_MAX_DIST, OBS_SIGMA0, OBS_SIGMA_PER_FT, PARALLEL_ALPHA,
                     PARALLEL_ANCHOR_ROWS, PARALLEL_COS, PARALLEL_DECAY, PARALLEL_FIT_GAIN,
                     PARALLEL_FIT_MAX, PARALLEL_MIN_PTS, PARALLEL_QUERY_K, PARALLEL_QUERY_STRIDE,
                     PARALLEL_RADIUS, PARALLEL_RES_MAX, PARALLEL_SHIFT_MAX, PARALLEL_SHIFT_MIN,
                     POLY_ALPHA, POLY_DEG, PROFILE_GRID, ROBUST_C, SD_CONF_POWER, SD_HEEL_ROWS,
                     SD_HEEL_TAU, SD_LEVEL_WINDOW, SD_MIN_ROWS, SD_NCC_WEIGHT, SD_NCC_WINDOW,
                     SD_OFFSET_PRIOR, SD_OFFSETS, SD_ROWS, SD_SMOOTH, SD_TEMPERATURE,
                     SD_TRANS_WIDTH, SHRINK_D0, SHRINK_K, SHRINK_MAX, SHRINK_SPAN_GUARD, SNAP_ALPHA,
                     SNAP_KNOTS, START_CONTINUITY, TYPEWELL_CORR_MIN)
from .particle_filter import typewell_curve


# --- Neighbour wells: marker surface and parallel-neighbour correction ----------------------------
# Cross-well information: the structural surface S = Z + TVT is shared by nearby wells.
#
# For a training well, TVT is known along its whole length, so each of its rows is a point
# (X, Y, S) on the marker surface. Interpolating that cloud at the target well's (X, Y) and
# subtracting the target's known Z gives an independent, GR-free TVT estimate. The target well is
# always excluded from its own cloud, so out-of-fold a well never sees its own labels this way.

log = logging.getLogger(__name__)


def _typewell_profile(tw: pd.DataFrame) -> np.ndarray:
    tt = tw["TVT"].values.astype(float)
    tg = tw["GR"].values.astype(float)
    o = np.argsort(tt)
    return np.interp(PROFILE_GRID, tt[o], tg[o], left=np.nan, right=np.nan)


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50: return -1.0
    aa = a[m] - a[m].mean()
    bb = b[m] - b[m].mean()
    d = np.sqrt((aa * aa).sum() * (bb * bb).sum())
    return float((aa * bb).sum() / d) if d > 0 else -1.0


def build_index(data_dir: str, strict: bool = True) -> dict:
    """Point clouds of the marker surface S = Z + TVT over all training wells.

    A training well that cannot be read raises, unless strict=False, in which case it is skipped
    with a warning.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "train", "*__horizontal_well.csv")))
    if not files:
        raise FileNotFoundError(
            f"no training wells (*__horizontal_well.csv) under "
            f"{os.path.join(data_dir, 'train')}"
        )
    (cloud_x, cloud_y, cloud_surface, cloud_well, dense_x, dense_y, dense_surface, dense_well,
     profiles, names, azimuths) = ([] for _ in range(11))
    for f in files:
        w = os.path.basename(f).split("__")[0]
        try:
            h = pd.read_csv(f)
            tw = pd.read_csv(os.path.join(data_dir, "train", f"{w}__typewell.csv"))
        except Exception as e:
            if strict: raise
            log.warning("skipping unreadable training well %s: %s", w, e)
            continue
        if "TVT" not in h.columns: continue
        T = h["TVT"].values.astype(float)
        Z = h["Z"].values.astype(float)
        X = h["X"].values.astype(float)
        Y = h["Y"].values.astype(float)
        m = np.isfinite(T) & np.isfinite(Z) & np.isfinite(X) & np.isfinite(Y)
        if m.sum() < MIN_CLOUD_ROWS: continue
        wi = len(names)
        idx = np.where(m)[0][::CLOUD_STRIDE]
        cloud_x.append(X[idx])
        cloud_y.append(Y[idx])
        cloud_surface.append(Z[idx] + T[idx])
        cloud_well.append(np.full(len(idx), wi))
        az = np.array([X[idx][-1] - X[idx][0], Y[idx][-1] - Y[idx][0]])
        azimuths.append(az / (np.hypot(*az) + 1e-9))
        idd = np.where(m)[0][::DENSE_STRIDE]
        dense_x.append(X[idd])
        dense_y.append(Y[idd])
        dense_surface.append(Z[idd] + T[idd])
        dense_well.append(np.full(len(idd), wi))
        profiles.append(_typewell_profile(tw))
        names.append(w)
    cloud_x, cloud_y, cloud_surface, cloud_well = (
        np.concatenate(a) for a in (cloud_x, cloud_y, cloud_surface, cloud_well)
    )
    dense_x, dense_y, dense_surface, dense_well = (
        np.concatenate(a) for a in (dense_x, dense_y, dense_surface, dense_well)
    )
    return dict(
        cloud_tree=cKDTree(np.c_[cloud_x, cloud_y]),
        cloud_x=cloud_x,
        cloud_y=cloud_y,
        cloud_surface=cloud_surface,
        cloud_well=cloud_well,
        dense_tree=cKDTree(np.c_[dense_x, dense_y]),
        dense_x=dense_x,
        dense_y=dense_y,
        dense_surface=dense_surface,
        dense_well=dense_well,
        profiles=np.array(profiles),
        azimuths=np.array(azimuths),
        well_id={w: i for i, w in enumerate(names)},
    )


def marker_transfer(h: pd.DataFrame, tw: pd.DataFrame, rows: np.ndarray, index: dict, name: str):
    """Neighbour-interpolated TVT along the eval rows, anchored to this well's last known TVT.

    Only neighbours whose typewell correlates with ours (same stratigraphic frame) are used.
    Along the lateral, neighbours are ranked by an anisotropic distance that discounts the
    cross-track direction: wells crossing our path sample the surface where we will be.
    Returns (TVT per eval row, nearest-neighbour distance per eval row) or (None, inf).
    """
    prof = _typewell_profile(tw)
    same_frame = np.array(
        [
            _corr(prof, index["profiles"][i]) > TYPEWELL_CORR_MIN
            for i in range(len(index["profiles"]))
        ]
    )
    self_id = index["well_id"].get(name, -1)
    if self_id >= 0: same_frame[self_id] = False
    if not same_frame.any(): return None, np.inf
    k_query = min(
        CLOUD_QUERY_K, len(index["cloud_surface"])
    )  # tiny datasets have fewer cloud points
    X = h["X"].values.astype(float)
    Y = h["Y"].values.astype(float)
    Z = h["Z"].values.astype(float)

    def isotropic(xs, ys, zs):
        d, idx = index["cloud_tree"].query(np.c_[xs, ys], k=k_query)
        src = index["cloud_well"][idx]
        out = np.full(len(xs), np.nan)
        for i in range(len(xs)):
            keep = [k for k in range(k_query) if same_frame[src[i, k]]][:ANCHOR_K]
            if not keep: continue
            wg = 1.0 / (d[i, keep] + 1e-3)
            out[i] = np.sum(wg * index["cloud_surface"][idx[i, keep]]) / wg.sum() - zs[i]
        return out

    def anisotropic(xs, ys, zs):
        d, idx = index["cloud_tree"].query(np.c_[xs, ys], k=k_query)
        src = index["cloud_well"][idx]
        tx = np.gradient(xs)
        ty = np.gradient(ys)
        tn = np.hypot(tx, ty)
        tn[tn == 0] = 1.0
        tx /= tn
        ty /= tn
        out = np.full(len(xs), np.nan)
        nd = np.full(len(xs), np.nan)
        for i in range(len(xs)):
            ok = same_frame[src[i]]
            dxm = index["cloud_x"][idx[i]] - xs[i]
            dym = index["cloud_y"][idx[i]] - ys[i]
            along = dxm * tx[i] + dym * ty[i]
            cross = -dxm * ty[i] + dym * tx[i]
            de = np.sqrt(along**2 + (CROSS_TRACK_SCALE * cross) ** 2)
            order = np.argsort(de)
            ki = order[ok[order]][:ANISO_K]
            if len(ki) == 0: continue
            wg = 1.0 / (de[ki].astype(float) + 1e-3)
            out[i] = np.sum(wg * index["cloud_surface"][idx[i, ki]]) / wg.sum() - zs[i]
            nd[i] = float(d[i, ki].astype(float).min())  # gating uses the true distance
        return out, nd

    ev = np.asarray(rows)
    query_pos = np.arange(0, len(ev), MARKER_QUERY_STRIDE)
    if query_pos[-1] != len(ev) - 1: query_pos = np.append(query_pos, len(ev) - 1)
    query_rows = ev[query_pos]
    tv, nd = anisotropic(X[query_rows], Y[query_rows], Z[query_rows])
    known = np.where(h["TVT_input"].notna().values)[0]
    if len(known) == 0: return None, np.inf
    anchor = known[-1]
    at = isotropic(np.array([X[anchor]]), np.array([Y[anchor]]), np.array([Z[anchor]]))
    shift = (float(h["TVT_input"].values[anchor]) - at[0]) if np.isfinite(at[0]) else 0.0
    m = np.isfinite(tv)
    if m.sum() < 3 or not np.isfinite(nd).any(): return None, np.inf
    full = np.interp(np.arange(len(ev)), query_pos[m], tv[m] + shift)
    nd_rows = np.interp(np.arange(len(ev)), query_pos[m], nd[m])
    return full, nd_rows


def marker_observations(tv, nd_rows):
    """Sparse marker observations for the filter: every OBS_EVERY rows where a neighbour is close.

    Returns (marker TVT, marker sigma) with sigma = 0 meaning "no observation", or (None, None).
    """
    if tv is None or nd_rows is None or np.ndim(nd_rows) == 0: return None, None
    n = len(tv)
    sel = np.zeros(n, bool)
    sel[::OBS_EVERY] = True
    okm = sel & (np.asarray(nd_rows) < OBS_MAX_DIST) & np.isfinite(tv)
    if not okm.any(): return None, None
    sigma = np.where(okm, OBS_SIGMA0 + OBS_SIGMA_PER_FT * np.asarray(nd_rows), 0.0)
    mk = np.where(okm, np.where(np.isfinite(tv), tv, 0.0), 0.0)
    return mk, sigma


def parallel_marker_transfer(h: pd.DataFrame, rows: np.ndarray, index: dict, name: str):
    """Datum from parallel neighbours: a weighted local plane fit of S = Z + TVT.

    Parallel laterals (same azimuth) are offset perpendicular to the track; a local plane absorbs
    that structural offset. Only interpolation (query inside the neighbour point hull) is allowed.
    Returns (TVT per eval row, plane-fit residual per eval row; inf where uncovered) or (None, inf).
    """
    self_id = index["well_id"].get(name, -1)
    X = h["X"].values.astype(float)
    Y = h["Y"].values.astype(float)
    Z = h["Z"].values.astype(float)
    ev = np.asarray(rows)
    track_dir = np.array([X[ev[-1]] - X[ev[0]], Y[ev[-1]] - Y[ev[0]]])
    track_dir = track_dir / (np.hypot(*track_dir) + 1e-9)
    well_dirs = index["azimuths"]

    def plane_at(xs, ys, zs):
        # never ask for more neighbours than the cloud holds (only matters on tiny datasets)
        d, idx = index["dense_tree"].query(
            np.c_[xs, ys], k=min(PARALLEL_QUERY_K, len(index["dense_surface"]))
        )
        out = np.full(len(xs), np.nan)
        res = np.full(len(xs), np.nan)
        for i in range(len(xs)):
            src = index["dense_well"][idx[i]]
            dd = d[i]
            keep = (
                (src != self_id)
                & (dd < PARALLEL_RADIUS)
                & (np.abs(well_dirs[src] @ track_dir) > PARALLEL_COS)
            )
            if keep.sum() < PARALLEL_MIN_PTS: continue
            Xn = index["dense_x"][idx[i][keep]]
            Yn = index["dense_y"][idx[i][keep]]
            Sn = index["dense_surface"][idx[i][keep]]
            dn = dd[keep]
            if not (Xn.min() < xs[i] < Xn.max() and Yn.min() < ys[i] < Yn.max()): continue
            wg = np.exp(-dn / PARALLEL_DECAY)
            A = np.c_[np.ones_like(Xn), Xn - xs[i], Yn - ys[i]]
            Wm = np.sqrt(wg)
            try:
                c, _, _, _ = np.linalg.lstsq(A * Wm[:, None], Sn * Wm, rcond=None)
            except Exception:
                continue
            out[i] = c[0] - zs[i]
            res[i] = np.sqrt(np.average((Sn - A @ c) ** 2, weights=wg))
        return out, res

    query_pos = np.arange(0, len(ev), PARALLEL_QUERY_STRIDE)
    if query_pos[-1] != len(ev) - 1: query_pos = np.append(query_pos, len(ev) - 1)
    query_rows = ev[query_pos]
    tv, res = plane_at(X[query_rows], Y[query_rows], Z[query_rows])
    m = np.isfinite(tv)
    if m.sum() < 3: return None, np.inf
    known = np.where(h["TVT_input"].notna().values)[0]
    if len(known) == 0: return None, np.inf
    anchor = known[-PARALLEL_ANCHOR_ROWS:]  # anchor to the last known rows
    atv, _ = plane_at(X[anchor], Y[anchor], Z[anchor])
    okh = np.isfinite(atv)
    shift = float(np.median(h["TVT_input"].values[anchor][okh] - atv[okh])) if okh.any() else 0.0
    full = np.interp(np.arange(len(ev)), query_pos[m], tv[m] + shift)
    res_rows = np.interp(np.arange(len(ev)), query_pos[m], res[m])
    cover = np.interp(np.arange(len(ev)), query_pos, m.astype(float))
    res_rows[cover < 0.5] = np.inf
    return full, res_rows


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    s = a.std() * b.std()
    return float((a * b).mean() / s) if s > 1e-9 else -1.0


def parallel_correction(
    h: pd.DataFrame, tw: pd.DataFrame, rows: np.ndarray, pred: np.ndarray, index: dict, name: str
) -> np.ndarray | None:
    """Shift a well toward its parallel neighbours' datum, only when that clearly helps the GR fit.

    Fires when (i) the parallel datum disagrees by a moderate, confidently-fitted offset, (ii) the
    current GR match is poor, and (iii) the shifted path matches GR much better. This targets wells
    that have locked onto a wrong cycle. Returns the corrected prediction or None (no change).
    """
    tvp, resp = parallel_marker_transfer(h, rows, index, name)
    if tvp is None: return None
    ok = np.isfinite(tvp) & np.isfinite(np.asarray(resp))
    if ok.sum() < 10: return None
    offset = float(np.median(pred[ok] - np.asarray(tvp)[ok]))
    if (
        not (PARALLEL_SHIFT_MIN < abs(offset) < PARALLEL_SHIFT_MAX)
        or float(np.median(np.asarray(resp)[ok])) >= PARALLEL_RES_MAX
    ):
        return None
    gr = h["GR"].values[np.asarray(rows)].astype(float)
    mg = np.isfinite(gr)
    if mg.sum() < 30: return None
    twt = tw["TVT"].values.astype(float)
    twg = tw["GR"].values.astype(float)
    fit_now = _ncc(gr[mg], np.interp(pred[mg], twt, twg))
    fit_shifted = _ncc(gr[mg], np.interp((pred - offset)[mg], twt, twg))
    if fit_now < PARALLEL_FIT_MAX and (fit_shifted - fit_now) > PARALLEL_FIT_GAIN:
        return pred - PARALLEL_ALPHA * offset
    return None


# --- EM-plane member ------------------------------------------------------------------------------
# EM-plane member: a grid HMM over TVT, guided by a fitted structural plane.
#
# Locally the formation is close to a plane, TVT ~ b . (X, Y, Z) + c, whose slopes give the expected
# TVT drift between HMM steps. Hard EM alternates: fit the plane to the heel plus the decoded path
# (ridge-pulled to the heel plane), then re-decode with Viterbi using that drift as the transition
# mean.

def _band_apply(v, kernel, offs):
    """Apply a banded transition kernel (state shift by each offset) to a probability vector."""
    out = np.zeros_like(v)
    S = len(v)
    for j, o in enumerate(offs):
        if o == 0:
            out += kernel[j] * v
        elif o > 0:
            out[o:] += kernel[j] * v[: S - o]
        else:
            out[: S + o] += kernel[j] * v[-o:]
    return out


def hmm_decode(well: dict, trans_sigma: float, decoder: str, k_dz: float = 0.0, drift=None):
    """Decode TVT on every EM_DECIM-th eval row; return TVT interpolated to all eval rows.

    decoder: "viterbi" (best path) or "marginal_mean" (forward-backward posterior mean).
    The transition mean per step is `drift` if given, else k_dz * dZ.
    """
    ev = well["ev"]
    DZ = EM_STATE_STEP
    idx = np.arange(0, len(ev), EM_DECIM)
    if idx[-1] != len(ev) - 1: idx = np.append(idx, len(ev) - 1)
    rows = ev[idx]
    md_r = well["MD"][rows]
    gr_r = well["gr"][rows]
    M = len(rows)
    states = well["last_tvt"] + np.arange(-EM_WINDOW[0], EM_WINDOW[1] + DZ, DZ)
    S = len(states)
    exp_gr = np.interp(states, well["twt"], well["twg"])
    log_emission = -0.5 * ((gr_r[:, None] - exp_gr[None, :]) / EM_GR_SIGMA) ** 2
    dMD = np.diff(md_r)
    dZ = np.diff(well["Z"][rows])
    offs = np.arange(-EM_BAND, EM_BAND + 1)
    logp0 = -0.5 * ((states - well["last_tvt"]) / EM_START_SIGMA) ** 2 + log_emission[0]

    def trans_log(k):
        dr = drift[k - 1] if drift is not None else k_dz * dZ[k - 1]
        return -0.5 * ((offs[None, :] - dr / DZ) / (trans_sigma * math.sqrt(dMD[k - 1]) / DZ)) ** 2

    if decoder == "viterbi":
        V = logp0.copy()
        back = np.empty((M, S), np.int32)
        sidx = np.arange(S)
        for k in range(1, M):
            T = trans_log(k)
            src = sidx[:, None] - offs[None, :]
            valid = (src >= 0) & (src < S)
            cand = np.where(valid, V[np.clip(src, 0, S - 1)] + T, -1e18)
            j = np.argmax(cand, axis=1)
            back[k] = np.clip(sidx - offs[j], 0, S - 1)
            V = cand[sidx, j] + log_emission[k]
        s = int(np.argmax(V))
        path = np.empty(M, np.int32)
        path[-1] = s
        for k in range(M - 1, 0, -1):
            s = back[k, s]
            path[k - 1] = s
        est = states[path]
    else:
        A = np.exp(logp0 - logp0.max())
        A /= A.sum()
        alpha = np.empty((M, S))
        alpha[0] = A
        Tk = []
        for k in range(1, M):
            tl = np.exp(trans_log(k))[0]
            tl /= tl.sum()
            Tk.append(tl)
            pred = _band_apply(alpha[k - 1], tl, offs)
            em = np.exp(log_emission[k] - log_emission[k].max())
            a = pred * em
            a /= a.sum() + 1e-300
            alpha[k] = a
        beta = np.empty((M, S))
        beta[-1] = 1.0 / S
        for k in range(M - 2, -1, -1):
            em = np.exp(log_emission[k + 1] - log_emission[k + 1].max())
            b = _band_apply(beta[k + 1] * em, Tk[k][::-1], offs)
            b /= b.sum() + 1e-300
            beta[k] = b
        post = alpha * beta
        post /= post.sum(axis=1, keepdims=True) + 1e-300
        est = post @ states
    return np.interp(np.arange(len(ev)), idx, est)


def emplane_trajectory(h: pd.DataFrame, tw: pd.DataFrame):
    """Return (eval row indices, EM-plane TVT per eval row) or (None, None)."""
    known = h["TVT_input"].notna().values
    ev = np.where(~known)[0]
    if len(ev) < 5 or known.sum() < 10: return None, None
    MD = h["MD"].values.astype(float)
    Z = h["Z"].values.astype(float)
    X = h["X"].values.astype(float)
    Y = h["Y"].values.astype(float)
    gr = h["GR"].values.astype(float)
    s = np.where(np.isfinite(gr))[0]
    if len(s) < 2: return None, None
    gr = np.interp(np.arange(len(gr)), s, gr[s])
    twt, twg = typewell_curve(tw)
    ki = np.where(known)[0]
    tvi = h["TVT_input"].values
    well = dict(MD=MD, gr=gr, twt=twt, twg=twg, ev=ev, Z=Z, last_tvt=float(tvi[ki[-1]]))

    # exploratory pass: how strongly does TVT follow Z here? (k = dTVT/dZ, clipped to EM_K_RANGE)
    p1 = hmm_decode(well, EM_TRANS_SIGMA[0], "marginal_mean")
    Zr = Z[ev]
    design = np.vstack([Zr - Zr.mean(), np.ones_like(Zr)]).T
    k_dz_fit = float(
        np.clip(np.linalg.lstsq(design, p1 - p1.mean(), rcond=None)[0][0], *EM_K_RANGE)
    )

    idx = np.arange(0, len(ev), EM_DECIM)
    if idx[-1] != len(ev) - 1: idx = np.append(idx, len(ev) - 1)
    step_rows = ev[idx]
    tail = ki[-EM_HEEL_ROWS:]
    plane_prior = np.linalg.lstsq(
        np.c_[X[tail], Y[tail], Z[tail], np.ones(len(tail))], tvi[tail], rcond=None
    )[0]
    p = hmm_decode(well, EM_TRANS_SIGMA[1], "viterbi", k_dz=k_dz_fit)
    ridge = np.diag([EM_RIDGE, EM_RIDGE, EM_RIDGE, 1.0])
    for _ in range(EM_ITERS):
        x_all = np.concatenate([X[ki], X[ev]])
        y_all = np.concatenate([Y[ki], Y[ev]])
        z_all = np.concatenate([Z[ki], Z[ev]])
        tvt_all = np.concatenate([tvi[ki], p])
        design_xyz = np.c_[x_all, y_all, z_all, np.ones(len(x_all))]
        b = np.linalg.solve(
            design_xyz.T @ design_xyz + ridge, design_xyz.T @ tvt_all + ridge @ plane_prior
        )
        drift = (
            b[0] * np.diff(X[step_rows])
            + b[1] * np.diff(Y[step_rows])
            + b[2] * np.diff(Z[step_rows])
        )
        p = hmm_decode(well, EM_TRANS_SIGMA[1], "viterbi", drift=drift)
    return ev, p


# --- Whole-well shape decoder ---------------------------------------------------------------------
# Whole-well shape decoder: a final HMM over a smooth offset added to the stacked prediction.
#
# Hidden state per sampled row: an offset o to the prediction p, scored by GR value and windowed
# correlation against the typewell at p + o. The applied offset is the posterior mean times
# confidence**3 (1 - normalised entropy), so the decoder acts only where the posterior is sharp.
# It is damped by exp(-heel_rms / tau) when the same decoder moves the known heel, where the true
# offset is zero.

# transition kernel: the offset changes smoothly along the well
_TRANS = np.exp(-SD_TRANS_WIDTH * (SD_OFFSETS[:, None] - SD_OFFSETS[None, :]) ** 2)
_TRANS /= _TRANS.sum(1, keepdims=True)
_S = len(SD_OFFSETS)


def _box(a, w):
    """Length-preserving windowed sum over axis 0 of an (n, S) array."""
    pad = w // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    cs = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(ap, 0)])
    return cs[w : w + a.shape[0]] - cs[: a.shape[0]]


def _emission_cost(g, E):
    """Cost (rows, offsets): local mean |GR - expected| + weight * (1 - windowed Pearson)."""
    S = E.shape[1]
    WR = SD_NCC_WINDOW
    level = _box(np.abs(g[:, None] - E), SD_LEVEL_WINDOW) / SD_LEVEL_WINDOW
    gw = _box(g[:, None] * np.ones((1, S)), WR)
    g2 = _box((g**2)[:, None] * np.ones((1, S)), WR)
    Ew = _box(E, WR)
    E2 = _box(E**2, WR)
    gE = _box(g[:, None] * E, WR)
    cov = gE / WR - (gw / WR) * (Ew / WR)
    sg = np.sqrt(np.maximum(g2 / WR - (gw / WR) ** 2, 1e-9))
    se = np.sqrt(np.maximum(E2 / WR - (Ew / WR) ** 2, 1e-9))
    return level + SD_NCC_WEIGHT * (1 - cov / (sg * se + 1e-9))


def _forward_backward(cost):
    """Posterior-mean offset and confidence (1 - normalised entropy) per row."""
    pot = np.exp(-(cost + SD_OFFSET_PRIOR * (SD_OFFSETS[None, :] ** 2)) / SD_TEMPERATURE)
    pot /= pot.sum(1, keepdims=True) + 1e-30
    r = cost.shape[0]
    al = np.empty((r, _S))
    al[0] = pot[0]
    for i in range(1, r):
        al[i] = pot[i] * (al[i - 1] @ _TRANS)
        al[i] /= al[i].sum() + 1e-30
    be = np.empty((r, _S))
    be[-1] = 1.0
    for i in range(r - 2, -1, -1):
        be[i] = (pot[i + 1] * be[i + 1]) @ _TRANS.T
        be[i] /= be[i].sum() + 1e-30
    post = al * be
    post /= post.sum(1, keepdims=True) + 1e-30
    ent = -(post * np.log(post + 1e-12)).sum(1) / np.log(_S)
    return post @ SD_OFFSETS, np.clip(1.0 - ent, 0, 1)


def shape_decode(h: pd.DataFrame, tw: pd.DataFrame, rows: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Refined prediction for `rows`; returns `p` itself (same object) when the decoder abstains."""
    rows = np.asarray(rows)
    twt, twg = typewell_curve(tw)
    if len(twt) < 5 or len(rows) < SD_MIN_ROWS: return p
    known = h["TVT_input"].notna().values
    gr_raw = h["GR"].values.astype(float)
    hidden = np.where(~known)[0]
    pos = np.searchsorted(hidden, rows)
    if pos.max() >= len(hidden) or not np.all(hidden[pos] == rows):
        gr_obs = pd.Series(gr_raw).interpolate(limit_direction="both").to_numpy()[rows]
    else:
        gr_obs = pd.Series(gr_raw[hidden]).interpolate(limit_direction="both").to_numpy()[pos]
    # affine map of measured GR onto the typewell scale, fitted on the known heel
    tvi = h["TVT_input"].values.astype(float)
    known_rows = np.where(known)[0]
    if len(known_rows) > 50:
        gr_known = pd.Series(gr_raw[known_rows]).interpolate(limit_direction="both").to_numpy()
        tw_known = np.interp(tvi[known_rows], twt, twg)
        valid = np.isfinite(gr_known) & np.isfinite(tw_known)
        if valid.sum() > 50:
            a, b = np.linalg.lstsq(
                np.vstack([gr_known[valid], np.ones(valid.sum())]).T, tw_known[valid], rcond=None
            )[0]
            gr_obs = gr_obs * a + b
    if not np.all(np.isfinite(gr_obs)): return p
    n = len(p)
    sample_rows = np.linspace(0, n - 1, min(n, SD_ROWS)).astype(int)
    E = np.interp(p[sample_rows][:, None] + SD_OFFSETS[None, :], twt, twg)
    offset_s, conf_s = _forward_backward(_emission_cost(gr_obs[sample_rows], E))
    heel_rms = 0.0
    if len(known_rows) > 60:
        heel_rows = known_rows[
            np.linspace(0, len(known_rows) - 1, min(len(known_rows), SD_HEEL_ROWS)).astype(int)
        ]
        gr_heel = pd.Series(gr_raw[heel_rows]).interpolate(limit_direction="both").to_numpy()
        gr_heel = (
            gr_heel - np.nanmedian(gr_heel) + np.nanmedian(np.interp(tvi[heel_rows], twt, twg))
        )
        expected_heel = np.interp(tvi[heel_rows][:, None] + SD_OFFSETS[None, :], twt, twg)
        offset_heel = _forward_backward(_emission_cost(gr_heel, expected_heel))[0]
        heel_rms = float(np.sqrt(np.mean((offset_heel - np.median(offset_heel)) ** 2)))
    trust = np.exp(-heel_rms / SD_HEEL_TAU)
    off = np.interp(np.arange(n), sample_rows, offset_s * (conf_s**SD_CONF_POWER) * trust)
    k = np.ones(SD_SMOOTH) / SD_SMOOTH
    half = SD_SMOOTH // 2
    off = np.convolve(np.pad(off, half, mode="edge"), k, "same")[half : half + n]
    return p + off


# --- Shape regulariser and final corrections ------------------------------------------------------
# Physically motivated corrections applied to a well's TVT prediction.

def _robust_reweight(y, fit):
    """Cauchy-style weights from residuals scaled by the MAD."""
    r = np.abs(y - fit)
    sd = max(np.median(r) * 1.4826, 1e-6)
    return 1.0 / np.sqrt(1.0 + (r / (ROBUST_C * sd)) ** 2)


def _robust_pwl(x, y, K):
    """Robust continuous piecewise-linear fit with K equal segments (iteratively reweighted)."""
    if len(x) < K + 2: return y.copy()
    knots = np.linspace(x[0], x[-1], K + 1)
    B = np.zeros((len(x), len(knots)))
    for j in range(len(knots)):  # hat basis functions
        lo = knots[j - 1] if j > 0 else knots[0] - 1
        hi = knots[j + 1] if j < len(knots) - 1 else knots[-1] + 1
        l = (x >= lo) & (x <= knots[j])
        B[l, j] = (x[l] - lo) / (knots[j] - lo + 1e-9)
        rr = (x > knots[j]) & (x <= hi)
        B[rr, j] = (hi - x[rr]) / (hi - knots[j] + 1e-9)
    wgt = np.ones(len(x))
    fit = y
    for _ in range(IRLS_ITERS):
        c, *_ = np.linalg.lstsq(B * wgt[:, None], y * wgt, rcond=None)
        fit = B @ c
        wgt = _robust_reweight(y, fit)
    return fit


def _robust_poly(x, y, deg):
    if len(x) < deg + 2: return y.copy()
    xn = (x - x.min()) / (np.ptp(x) + 1e-9)
    wgt = np.ones(len(x))
    fit = y
    for _ in range(IRLS_ITERS):
        coef = np.polyfit(xn, y, deg, w=wgt)
        fit = np.polyval(coef, xn)
        wgt = _robust_reweight(y, fit)
    return fit


def snap(pred: np.ndarray, md: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Regularise the datum u = TVT + Z toward a robust piecewise-linear, then cubic, shape.

    Real formations hold a dip for long stretches, so u is close to piecewise linear in MD;
    the fit removes filter jitter while the known Z keeps all fast TVT wiggles.
    """
    if len(pred) < SNAP_KNOTS + 2: return pred
    snapped = _robust_pwl(md, pred + z, SNAP_KNOTS) - z
    u = (1 - SNAP_ALPHA) * pred + SNAP_ALPHA * snapped + z
    u = (1 - POLY_ALPHA) * u + POLY_ALPHA * _robust_poly(md, u, POLY_DEG)
    return u - z


def shape_regularize(drift: np.ndarray, t0: float, md: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Snap a filter drift to a plausible shape, then shrink toward the heel TVT if uncertain.

    A large disagreement between raw and snapped paths means the filter wandered; such wells are
    shrunk a little toward t0 (bounded at 25%), unless the path spans a large TVT range for real.
    """
    raw = t0 + drift
    ps = snap(raw, md, z)
    dis = float(np.abs(ps - raw).mean())
    alpha = float(np.clip(SHRINK_K * (dis - SHRINK_D0), 0.0, SHRINK_MAX))
    if float(np.ptp(ps)) > SHRINK_SPAN_GUARD: alpha = 0.0
    return ps if alpha == 0.0 else (1 - alpha) * ps + alpha * t0


def heel_taper(h: pd.DataFrame, rows: np.ndarray, p: np.ndarray) -> np.ndarray | None:
    """Remove a physically impossible jump at the hand-off from known to predicted TVT.

    If the first predicted row implies a dip steeper than HEEL_MAX_DIP from the last known row,
    the excess is removed and tapered linearly to zero over HEEL_TAPER_ROWS rows.
    Returns the corrected prediction or None (no change).
    """
    tvi = h["TVT_input"].values
    kn = np.where(np.isfinite(tvi))[0]
    if len(kn) == 0: return None
    md = h["MD"].values.astype(float)
    rr = np.asarray(rows)
    if rr.max() >= len(md): return None
    heel = float(tvi[kn[-1]])
    md_heel = float(md[kn[-1]])
    cap = abs(HEEL_MAX_DIP) * max(md[rr[0]] - md_heel, 1e-6)
    jump = p[0] - heel
    if abs(jump) <= cap: return None
    shift = jump - np.sign(jump) * cap
    K = min(HEEL_TAPER_ROWS, len(p))
    p = p.copy()
    p[:K] = p[:K] - shift * np.linspace(1.0, 0.0, K)
    return p


def start_continuity(h: pd.DataFrame, p: np.ndarray) -> np.ndarray | None:
    """Close part of the gap between the first prediction and the last known TVT.

    The filter starts its particles scattered around a datum known exactly, which leaves a small
    discontinuity at the hand-off (true TVT is continuous there). A fixed fraction of that gap is
    removed from the whole well.
    """
    tvi = h["TVT_input"].values
    kn = np.where(np.isfinite(tvi))[0]
    if not len(kn): return None
    gap = float(p[0]) - float(tvi[kn[-1]])
    return p - START_CONTINUITY * gap
