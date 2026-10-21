"""All model constants of the pipeline, in one place. Numerical guards (tiny epsilons, NaN
fallbacks) stay inline in the code.

Units: depths and distances in feet, gamma ray (GR) in API units, rates in ft/ft.
Values were tuned during the competition, mostly by self-excluded out-of-fold validation on the
773 training wells.
"""

import numpy as np

# --- Particle filter: state = (datum, rate), datum = TVT + Z ------------------------------
N_SEEDS = 128  # independent filter runs per well, averaged by likelihood
N_PARTICLES = 300  # particles per run
SEED_TEMPERATURE = 2.0  # softmax temperature on per-run log-likelihood (higher = hedge more)
GRID_STEP = 0.5  # resolution of the reference GR(TVT) curve
GR_SIGMA = 45.0  # Gaussian scale of the GR-level emission
FLANK_WEIGHT = 1.0  # weight of the GR-gradient (flank) emission relative to the level term
INIT_POS_SD = 2.0  # initial datum scatter around the last known point
INIT_RATE_SD = 0.01  # initial dip-rate scatter around the heel dip
RATE_MOMENTUM = 0.998  # AR(1) persistence of the structural dip rate per step
DIP_PERSISTENCE = 0.6  # the rate reverts to this fraction of the heel dip, not to zero
RATE_NOISE = 0.002  # per-step random walk of the dip rate
POS_NOISE = 0.005  # per-step jitter of the datum
RESAMPLE_ESS = 0.5  # resample when effective sample size < this fraction of particles
RESAMPLE_POS_JITTER = 0.1  # jitter added to resampled datums (keeps diversity)
RESAMPLE_RATE_JITTER = 0.001  # jitter added to resampled rates
TVT_MARGIN = 100.0  # particles are clamped to the typewell span +- this margin
HEEL_DIP_ROWS = 30  # last known rows used to estimate the heel dip
OWN_REF_MIN_ROWS = 50  # known rows needed to build the well's own GR(TVT) reference
OWN_REF_GATE_MARGIN = 0.001  # own reference must beat the typewell's per-row log-lik by this
OWN_REF_MIN_COVER = 20  # grid points the heel must cover for an own reference
LOGLIK_FLOOR = -300.0  # per-particle log-likelihood below this counts as zero likelihood
CHI2_CUTOFF = 600.0  # squared z-score beyond which an emission is treated as zero
LIK_FLOOR = 1e-300  # smallest particle likelihood (keeps weights positive)
MARKER_TAIL_LIK = 1e-12  # likelihood of a particle far outside a marker observation

# --- Mixture of decoders -----------------------------------------------------------------
MIX_FILTER, MIX_SMOOTHER, MIX_MARKER = 0.1, 0.6, 0.3  # filter / smoother / marker-observed filter
SMOOTH_BETA = MIX_SMOOTHER / (MIX_FILTER + MIX_SMOOTHER)  # used when no marker arm exists

# --- Shape regularizer (piecewise-linear + cubic fit of the datum) ------------------------
SNAP_KNOTS, SNAP_ALPHA = 16, 0.6  # 16-segment robust PWL fit, blended 60%
POLY_DEG, POLY_ALPHA = 3, 0.5  # robust cubic fit, blended 50%
SHRINK_D0, SHRINK_K, SHRINK_MAX = 1.0, 0.3, 0.25  # shrink toward heel TVT when the fit disagrees
SHRINK_SPAN_GUARD = 55.0  # never shrink wells with a large real TVT span
ROBUST_C = 2.0  # Cauchy weight scale, in robust (MAD) standard deviations
IRLS_ITERS = 4  # reweighting iterations of the robust fits

# --- Neighbour marker surface (S = Z + TVT is a geological surface shared by nearby wells) -
PROFILE_GRID = np.arange(9000.0, 14000.0, 2.0)  # common TVT grid for typewell similarity
TYPEWELL_CORR_MIN = 0.85  # neighbours must share the same stratigraphic frame
MIN_CLOUD_ROWS = 100  # a training well needs this many labelled rows to join the cloud
CLOUD_STRIDE = 25  # subsample of neighbour points for the marker cloud
CLOUD_QUERY_K = 60  # nearest cloud points examined per query
DENSE_STRIDE = 6  # denser cloud for local plane fits
ANCHOR_K = 45  # neighbours averaged at the heel anchor point
ANISO_K = 12  # neighbours averaged along the lateral
CROSS_TRACK_SCALE = 0.4  # cross-track distance discount (cross-azimuth neighbours preferred)
MARKER_QUERY_STRIDE = 25  # evaluate the marker surface every N eval rows

# marker as a sparse observation inside the filter
OBS_EVERY = 300  # one marker observation every N eval rows
OBS_SIGMA0 = 10.0  # observation sigma at zero neighbour distance
OBS_SIGMA_PER_FT = 0.02  # sigma growth per ft of neighbour distance
OBS_MAX_DIST = 1500.0  # no observation beyond this neighbour distance

# --- EM-plane member: HMM over TVT states + a structural plane TVT ~ (X, Y, Z) -------------
EM_STATE_STEP = 2.5  # TVT state spacing
EM_WINDOW = (600.0, 400.0)  # state window below / above the heel TVT
EM_DECIM = 50  # HMM steps every N eval rows
EM_BAND = 80  # max state jump per step (in states)
EM_GR_SIGMA = 45.0  # GR emission scale
EM_START_SIGMA = 3.0  # prior width around the heel TVT
EM_TRANS_SIGMA = (1.5, 0.5)  # transition sigma per sqrt(ft): exploratory pass / plane-guided passes
EM_ITERS = 2  # plane <-> Viterbi alternations
EM_HEEL_ROWS = 200  # last known rows used for the prior plane
EM_RIDGE = 1e3  # ridge pull of the plane slopes toward the heel-fitted plane
EM_K_RANGE = (-0.5, 0)  # allowed dTVT/dZ coupling for the exploratory transition drift
EM_KNOT_STRIDE = 8  # the member is linearly interpolated between every 8th eval row

# --- Whole-well shape decoder (HMM over a smooth offset to the stacked prediction) ---------
SD_OFFSETS = np.arange(-40, 40.01, 0.5)  # candidate offsets
SD_ROWS = 220  # rows sampled along the well
SD_MIN_ROWS = 60  # shorter wells are not decoded
SD_LEVEL_WINDOW = 5  # window of the |GR - expected| level cost
SD_TRANS_WIDTH = 1.0  # transition kernel exp(-width * (offset step)^2)
SD_NCC_WINDOW = 15  # window of the GR shape correlation
SD_NCC_WEIGHT = 3.0  # weight of (1 - correlation) in the emission cost
SD_OFFSET_PRIOR = 0.03  # quadratic pull toward offset 0 (stay near the stacked prediction)
SD_TEMPERATURE = 6.0  # cost -> probability temperature
SD_CONF_POWER = 3.0  # offset applied = posterior mean * confidence ** power
SD_HEEL_TAU = 8.0  # trust = exp(-heel decoding error / tau)
SD_HEEL_ROWS = 150  # known heel rows sampled for the self-check
SD_SMOOTH = 251  # moving-average window of the applied offset

# --- Stacking with the marker surface and the EM-plane member ----------------------------
MARKER_SCALE = 0.80  # global scale on the marker blend weight
MARKER_A0, MARKER_DIS0, MARKER_AMAX = 0.6, 60.0, 0.9  # weight = A0*exp(-disagreement/DIS0), capped
MARKER_DIST0 = 500.0  # marker weight decays with neighbour distance
AGREE_BOOST = 0.6  # boost when marker and EM-plane agree on the correction
EMPLANE_B0, EMPLANE_DIST0 = 0.25, 650.0  # EM-plane weight grows where neighbours are far
EMPLANE_BMAX = 0.9  # cap on the EM-plane weight after the agreement boost

# --- Parallel-neighbour datum correction ---------------------------------------------------
PARALLEL_COS = 0.94  # |cos(azimuth difference)| above this = parallel lateral (~20 deg)
PARALLEL_RADIUS = 600.0  # neighbour search radius
PARALLEL_QUERY_K = 250  # dense-cloud points examined per plane fit
PARALLEL_QUERY_STRIDE = 20  # fit the parallel plane every N eval rows
PARALLEL_ANCHOR_ROWS = 15  # last known rows used to anchor the parallel datum
PARALLEL_MIN_PTS = 8  # points needed for a local plane fit
PARALLEL_DECAY = 250.0  # distance decay of plane-fit weights
PARALLEL_SHIFT_MIN, PARALLEL_SHIFT_MAX = 3.0, 16.0  # only correct offsets in this band
PARALLEL_RES_MAX = 2.0  # plane-fit residual must be below this
PARALLEL_FIT_MAX = 0.60  # only correct wells whose GR match is currently poor (NCC)
PARALLEL_FIT_GAIN = 0.20  # and where the shifted path improves the NCC by this much
PARALLEL_ALPHA = 0.3  # fraction of the offset applied

# --- Final touches ------------------------------------------------------------------------
HEEL_MAX_DIP, HEEL_TAPER_ROWS = 0.10, 1000  # max physical dip at hand-off, taper length
START_CONTINUITY = 0.25  # fraction of the hand-off gap removed from the whole prediction
