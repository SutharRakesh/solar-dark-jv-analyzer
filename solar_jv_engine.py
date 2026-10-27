"""Scientific calculation engine shared with the solar-cell dark J-V desktop version.

This module intentionally has NO Qt, PySide6, pyqtgraph, or Streamlit imports.
Unmodified physics and fitting functions are extracted from the supplied desktop version.
"""

import io
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import wrightomega

K_B = 1.380649e-23
Q_E = 1.602176634e-19

# Wide NUMERICAL bounds, not universal physical acceptance criteria.
# Modify these for a known solar-cell technology / measurement setup.
PARAM_BOUNDS = {
    "J0": (1e-20, 1e-1), "J01": (1e-20, 1e-1), "J02": (1e-20, 1e-1),
    "n": (0.5, 6.0), "n1": (0.5, 6.0), "n2": (0.5, 6.0),
    "rs": (1e-10, 1e6), "rsh": (1e-1, 1e14),
}
LOG_KEYS = {"J0", "J01", "J02", "rs", "rsh"}
SINGLE_KEYS = ("J0", "n", "rs", "rsh")
TWO_KEYS = ("J01", "n1", "J02", "n2", "rs", "rsh")


@dataclass
class EstimateResult:
    J0: float = np.nan
    n: float = np.nan
    rs_area: float = np.nan
    rsh_area: float = np.nan
    rsh_r2: float = np.nan
    diode_r2: float = np.nan
    rs_r2: float = np.nan
    rsh_points: int = 0
    diode_points: int = 0
    rs_points: int = 0
    warnings: list = field(default_factory=list)


def thermal_voltage(T):
    return K_B * float(T) / Q_E


def linear_r2(x, y, slope, intercept):
    pred = slope * np.asarray(x) + intercept
    yy = np.asarray(y)
    denom = np.sum((yy - np.mean(yy)) ** 2)
    return float(1 - np.sum((yy - pred) ** 2) / denom) if denom > 0 else np.nan


def parse_two_columns(text):
    """Preserve ALL numeric data in acquisition order; do not average or sort."""
    for kwargs in (
        dict(sep=None, engine="python"),
        dict(sep=r"\s+", engine="python"),
        dict(sep=",", engine="python"),
        dict(sep="\t", engine="python"),
    ):
        try:
            df = pd.read_csv(io.StringIO(text), comment="#", header=None, **kwargs)
            if df.shape[1] < 2:
                continue
            out = df.iloc[:, :2].copy()
            out.columns = ["V_raw", "Y_raw"]
            out["V_raw"] = pd.to_numeric(out["V_raw"], errors="coerce")
            out["Y_raw"] = pd.to_numeric(out["Y_raw"], errors="coerce")
            out = out.dropna().reset_index(drop=True)
            if len(out) >= 5:
                return out
        except Exception:
            continue
    raise ValueError("Paste at least five numeric rows with voltage and J/I columns.")


def raw_to_internal(V_raw, Y_raw, input_mode, area_cm2,
                    flip_voltage=False, flip_current=False):
    V = np.asarray(V_raw, dtype=float) * (-1 if flip_voltage else 1)
    Y = np.asarray(Y_raw, dtype=float)
    if input_mode == "V + J (mA/cm^2)":
        J = Y / 1000.0
    elif input_mode == "V + J (A/cm^2)":
        J = Y.copy()
    elif input_mode == "V + I (A)":
        if area_cm2 <= 0:
            raise ValueError("Area must be positive.")
        J = Y / area_cm2
    else:
        raise ValueError("Unknown input mode")
    J = J * (-1 if flip_current else 1)
    finite = np.isfinite(V) & np.isfinite(J)
    return V[finite], J[finite], finite


def internal_fit_to_raw_units(V_internal, J_internal, input_mode, area_cm2,
                              flip_voltage=False, flip_current=False):
    V = np.asarray(V_internal, dtype=float) * (-1 if flip_voltage else 1)
    J = np.asarray(J_internal, dtype=float) * (-1 if flip_current else 1)
    if input_mode == "V + J (mA/cm^2)":
        Y = J * 1000.0
    elif input_mode == "V + J (A/cm^2)":
        Y = J
    elif input_mode == "V + I (A)":
        Y = J * area_cm2
    else:
        raise ValueError("Unknown input mode")
    return V, Y


def raw_y_label(input_mode):
    if input_mode == "V + J (mA/cm^2)":
        return "J_raw_mA_cm2", "J_fit_mA_cm2"
    if input_mode == "V + J (A/cm^2)":
        return "J_raw_A_cm2", "J_fit_A_cm2"
    return "I_raw_A", "I_fit_A"


def region_mask(V, region, minimum=3):
    lo, hi = sorted(region)
    mask = np.isfinite(V) & (V >= lo) & (V <= hi)
    return mask if int(np.sum(mask)) >= minimum else None


# =============================================================
# PHYSICAL CURRENT SOLVERS
# =============================================================

def _junction_current(U, J0s, ns, rsh, T):
    """Current as a function of the *internal* junction voltage U.

    Clip large arguments only during bracketing (at roots of realistic
    solar-cell data the arguments are well below 120). The solver checks
    the computed solution against the UNCLIPPED physical equation.
    """
    Vt = thermal_voltage(T)
    val = np.asarray(U, dtype=float) / rsh
    deriv = np.full_like(np.asarray(U, dtype=float), 1.0 / rsh)
    for J0, n in zip(J0s, ns):
        x = np.clip(np.asarray(U, dtype=float) / (n * Vt), -120, 120)
        ex = np.exp(x)
        val = val + J0 * np.expm1(x)
        deriv = deriv + J0 * ex / (n * Vt)
    return val, deriv


def solve_j_single(V, J0, n, rs, rsh, T, J_seed=None):
    """Stable Wright-omega solution with safeguarded implicit fallback.

    In extreme cases (very large rs*J0) explicit terms nearly cancel.
    Those points are recalculated by solving the junction-voltage equation.
    J_seed is intentionally ignored: prediction is independent of measured J.
    """
    V = np.asarray(V, dtype=float)
    shape = V.shape
    v = V.reshape(-1)
    if not all(z > 0 for z in (J0, n, rs, rsh, T)):
        raise ValueError("Single-diode parameters must be positive")
    a = n * thermal_voltage(T)
    A = 1.0 + rs / rsh
    logz = np.log(rs * J0 / (A * a)) + (v + rs * J0) / (A * a)
    omega = wrightomega(logz)
    j = (v / rsh - J0) / A + (a / rs) * omega

    # Recheck against the ORIGINAL equation, not a clipped exponential.
    u = v - j * rs
    with np.errstate(over="ignore", invalid="ignore"):
        rhs = J0 * np.expm1(u / a) + u / rsh
    failure = (~np.isfinite(j) | ~np.isfinite(rhs) |
               (np.abs(j-rhs) > np.maximum(1e-13, 1e-8*np.abs(j))))
    if np.any(failure):
        j[failure] = _solve_j_implicit(v[failure], (J0,), (n,), rs, rsh, T)
    return j.reshape(shape)


def _solve_j_implicit(V, J0s, ns, rs, rsh, T):
    """Passive single/two-diode junction-voltage solver with unique root."""
    V = np.asarray(V, dtype=float)
    shape = V.shape
    v = V.reshape(-1)
    if not all(z > 0 for z in (*J0s, *ns, rs, rsh, T)):
        raise ValueError("Diode parameters must be positive")
    lo = np.minimum(v, 0.0).copy()
    hi = np.maximum(v, 0.0).copy()
    u = v / (1.0 + rs / rsh)
    if np.any(v > 0):
        for J0, n in zip(J0s, ns):
            up = n * thermal_voltage(T) * np.log1p(np.maximum(v, 0) / (rs * J0))
            u = np.where(v > 0, np.minimum(u, up), u)
    u = np.clip(u, lo, hi)

    def func(z):
        j, dj = _junction_current(z, J0s, ns, rsh, T)
        return z + rs * j - v, 1.0 + rs * dj

    for _ in range(110):
        f, fp = func(u)
        lo = np.where(f < 0, u, lo)
        hi = np.where(f > 0, u, hi)
        nxt = u - f / fp
        midpoint = lo + 0.5 * (hi - lo)
        bad = (~np.isfinite(nxt)) | (nxt <= lo) | (nxt >= hi)
        nxt = np.where(bad, midpoint, nxt)
        if np.all(np.abs(nxt - u) <= 2e-14 + 2e-13 * np.abs(v)):
            u = nxt
            break
        u = nxt

    f, _ = func(u)
    tol = 3e-12 * (1.0 + np.abs(v))
    bad = ~np.isfinite(f) | (np.abs(f) > tol)
    for i in np.flatnonzero(bad):
        vi = float(v[i])
        if vi == 0:
            u[i] = 0.0
            continue
        def scalar_f(z):
            jj, _ = _junction_current(np.array([z]), J0s, ns, rsh, T)
            return float(z + rs * jj[0] - vi)
        u[i] = brentq(scalar_f, min(vi, 0.0), max(vi, 0.0),
                      xtol=1e-14, rtol=9e-16, maxiter=200)
    j, _ = _junction_current(u, J0s, ns, rsh, T)
    if not np.all(np.isfinite(j)):
        raise FloatingPointError("Nonfinite implicit diode current")
    return j.reshape(shape)


def solve_j_two(V, J01, n1, J02, n2, rs, rsh, T, J_seed=None):
    """Safeguarded unique-root solver; independent of acquisition order."""
    return _solve_j_implicit(V, (J01, J02), (n1, n2), rs, rsh, T)


def verify_physical_equation(V, J, params, T, two=False):
    """Relative closure error of the ORIGINAL, implicit dark-diode law."""
    U = np.asarray(V) - np.asarray(J) * params["rs"]
    j_terms = []
    pairs = (("J01", "n1"), ("J02", "n2")) if two else (("J0", "n"),)
    for jk, nk in pairs:
        x = U / (params[nk] * thermal_voltage(T))
        # A result at exponent > 600 cannot be trusted in float64.
        if np.any(x > 600):
            return np.inf
        j_terms.append(params[jk] * np.expm1(np.maximum(x, -700)))
    rhs = np.sum(j_terms, axis=0) + U / params["rsh"]
    scale = np.maximum.reduce([
        np.full_like(U, 1e-15), np.abs(J), np.abs(rhs), np.abs(U / params["rsh"])
    ] + [np.abs(z) for z in j_terms])
    return float(np.max(np.abs(np.asarray(J) - rhs) / scale))


# =============================================================
# PHYSICS-BASED INITIAL REGIONAL ESTIMATES
# =============================================================

def estimate_rsh(V, J, region):
    mask = region_mask(V, region, 4)
    if mask is None:
        return np.nan, np.nan, 0
    x, y = V[mask], J[mask]
    if np.ptp(x) < 1e-9:
        return np.nan, np.nan, len(x)
    slope, intercept = np.polyfit(x, y, 1)
    r2 = linear_r2(x, y, slope, intercept)
    # DO NOT use abs: negative conductance violates this passive model.
    value = 1.0 / slope if slope > 0 else np.nan
    return float(value), float(r2), len(x)


def estimate_diode(V, J, region, rsh, rs, T):
    mask = region_mask(V, region, 5)
    if mask is None:
        return np.nan, np.nan, np.nan, 0
    u = V - J * rs
    jd = J - u / rsh
    mask &= (V > 0) & (u > 0) & (jd > 0) & np.isfinite(jd)
    if np.sum(mask) < 5:
        return np.nan, np.nan, np.nan, int(np.sum(mask))
    x = np.log(jd[mask])
    y = u[mask]
    if np.ptp(x) < 0.3:
        return np.nan, np.nan, np.nan, len(x)
    m, b = np.polyfit(x, y, 1)
    n = m / thermal_voltage(T)
    r2 = linear_r2(x, y, m, b)
    if not np.isfinite(n) or n <= 0:
        return np.nan, np.nan, r2, len(x)
    ln_J0 = -b / m
    J0 = np.exp(ln_J0) if -700 <= ln_J0 <= 100 else np.nan
    return float(J0), float(n), float(r2), len(x)


def estimate_rs(V, J, region, rsh, n, T):
    mask = region_mask(V, region, 5)
    if mask is None or not np.isfinite(n) or n <= 0:
        return np.nan, np.nan, 0
    # First-pass shunt correction; refine Jd with estimated rs in later step.
    jd = J - V / rsh
    mask &= (V > 0) & (J > 0) & (jd > 0) & np.isfinite(jd)
    if np.sum(mask) < 5 or np.ptp(J[mask]) <= 0:
        return np.nan, np.nan, int(np.sum(mask))
    # In strong forward conduction: V = const + n Vt ln(Jd) + J*rs.
    corrected_v = V[mask] - n * thermal_voltage(T) * np.log(jd[mask])
    slope, intercept = np.polyfit(J[mask], corrected_v, 1)
    r2 = linear_r2(J[mask], corrected_v, slope, intercept)
    return (float(slope) if slope > 0 else np.nan), float(r2), int(np.sum(mask))


def estimate_all(V, J, T, rsh_region, diode_region, rs_region):
    e = EstimateResult()
    e.rsh_area, e.rsh_r2, e.rsh_points = estimate_rsh(V, J, rsh_region)
    rsh = e.rsh_area if np.isfinite(e.rsh_area) else 1e7
    if not np.isfinite(e.rsh_area):
        e.warnings.append("Rsh regional slope is unusable; a fallback start is used.")
    e.J0, e.n, e.diode_r2, e.diode_points = estimate_diode(
        V, J, diode_region, rsh, 0.0, T)
    n_guess = e.n if np.isfinite(e.n) and e.n > 0 else 1.5
    e.rs_area, e.rs_r2, e.rs_points = estimate_rs(
        V, J, rs_region, rsh, n_guess, T)
    rs_guess = e.rs_area if np.isfinite(e.rs_area) else 0.0
    # Second pass: use the series-voltage correction in the diode region.
    j0b, nb, r2b, pointsb = estimate_diode(
        V, J, diode_region, rsh, rs_guess, T)
    if np.isfinite(j0b) and np.isfinite(nb):
        e.J0, e.n, e.diode_r2, e.diode_points = j0b, nb, r2b, pointsb
    if np.isfinite(e.n):
        rsb, rsr2b, npts = estimate_rs(V, J, rs_region, rsh, e.n, T)
        if np.isfinite(rsb):
            e.rs_area, e.rs_r2, e.rs_points = rsb, rsr2b, npts
    if np.isfinite(e.n) and not 0.5 <= e.n <= 6:
        e.warnings.append("Regional ideality factor is outside fit bounds.")
    if np.isfinite(e.rsh_r2) and e.rsh_r2 < 0.95:
        e.warnings.append("Rsh region is not strongly linear; leakage may be non-ohmic.")
    if np.isfinite(e.diode_r2) and e.diode_r2 < 0.98:
        e.warnings.append("Diode region is not strongly exponential.")
    return e


def _candidate_windows(idx, min_count=7):
    """Candidate contiguous windows of SORTED indices for REGION SUGGESTION only."""
    N = len(idx)
    if N < min_count:
        return []
    sizes = sorted(set(min(N, max(min_count, z)) for z in
                       (min_count, 10, 14, 20, 30, N // 2, N)))
    result = []
    for size in sizes:
        stride = max(1, size // 5)
        starts = list(range(0, N - size + 1, stride))
        if not starts or starts[-1] != N - size:
            starts.append(N - size)
        for start in starts:
            result.append(idx[start:start + size])
    return result


def measurement_warnings(V):
    """Flags mixed forward/reverse scans without changing original data."""
    V = np.asarray(V, dtype=float)
    span = max(float(np.ptp(V)), 1e-12)
    steps = np.diff(V)
    dirs = np.sign(steps[np.abs(steps) > 1e-6 * span])
    turns = int(np.sum(dirs[1:] != dirs[:-1])) if len(dirs) >= 2 else 0
    return (["Voltage sweep reverses direction: do not fit hysteretic "
             "branches together; select one branch if currents differ."]
            if turns > 0 else [])


def suggest_regions(V, J, T):
    """Conservative suggestions; keep measured data and acquisition order.

    These are heuristics based on near-zero linearity, exponential behavior,
    and high-current behavior. The true 'best' region cannot be guaranteed
    from J-V alone, and must be inspected on the graph.
    """
    V, J = np.asarray(V), np.asarray(J)
    order = np.argsort(V, kind="stable")
    v, j = V[order], J[order]
    vmin, vmax = float(v[0]), float(v[-1])
    span = max(vmax - vmin, 1e-8)
    warnings = measurement_warnings(V)

    # Rsh candidate: prefer nearby reverse bias; if no reverse data,
    # use the lowest-voltage segment only as a starting point.
    if np.any(v <= 0) and np.any(v >= -0.25 * span):
        near = (v >= max(vmin, -max(0.15, 0.25 * span))) & (
            v <= min(vmax, max(0.025, 0.06 * span)))
    else:
        near = v <= vmin + 0.20 * span
        warnings.append("No useful reverse/zero-bias region; Rsh may be weakly constrained.")
    indices = np.flatnonzero(near)
    best = None
    for ids in _candidate_windows(indices, 6):
        xx, yy = v[ids], j[ids]
        if np.ptp(xx) < 0.005:
            continue
        m, b = np.polyfit(xx, yy, 1)
        if not np.isfinite(m) or m <= 0:
            continue
        r2 = linear_r2(xx, yy, m, b)
        split = len(ids) // 2
        m1 = np.polyfit(xx[:split], yy[:split], 1)[0]
        m2 = np.polyfit(xx[split:], yy[split:], 1)[0]
        curvature = abs(m1 - m2) / max(abs(m), 1e-30)
        score = r2 + 0.025 * np.ptp(xx) / max(np.ptp(v[indices]), 1e-12)
        score -= 0.035 * min(curvature, 10) + 0.01 * abs(np.mean(xx)) / span
        if best is None or score > best[0]:
            best = (score, (float(xx[0]), float(xx[-1])), r2)
    rsh_region = best[1] if best else (
        float(v[indices[0]]), float(v[indices[-1]])) if len(indices) >= 2 else (vmin, vmin + 0.15*span)
    if best is None or best[2] < 0.96:
        warnings.append("No clearly ohmic Rsh region identified; inspect manually.")

    rsh_guess, _, _ = estimate_rsh(V, J, rsh_region)
    rsh_eff = rsh_guess if np.isfinite(rsh_guess) else 1e7
    jd = j - v / rsh_eff
    forward = (v > 0) & (jd > 0) & (j > 0)
    if np.any(forward):
        largest = np.max(jd[forward])
        forward &= jd >= max(1e-15, largest * 1e-9)
    didx = np.flatnonzero(forward)
    d_best = None
    for ids in _candidate_windows(didx, 7):
        xx, yy = np.log(jd[ids]), v[ids]
        decades = np.ptp(xx) / np.log(10)
        if decades < 0.55 or np.ptp(yy) < 0.035:
            continue
        m, b = np.polyfit(xx, yy, 1)
        n = m / thermal_voltage(T)
        if not 0.55 <= n <= 5.5:
            continue
        r2 = linear_r2(xx, yy, m, b)
        # Prefer long, well-linear exponential intervals, not the highest J.
        score = r2 + 0.025 * min(decades, 4.0) + 0.0005 * len(ids)
        score -= 0.025 * np.median(jd[ids]) / max(np.max(jd[didx]), 1e-30)
        if d_best is None or score > d_best[0]:
            d_best = (score, (float(yy[0]), float(yy[-1])), r2)
    if d_best is not None:
        diode_region = d_best[1]
        if d_best[2] < 0.985:
            warnings.append("Exponential region shows curvature; n/J0 may be uncertain.")
    elif len(didx) >= 5:
        lo_i, hi_i = int(0.2*(len(didx)-1)), int(0.7*(len(didx)-1))
        diode_region = (float(v[didx[lo_i]]), float(v[didx[max(hi_i, lo_i+1)]]))
        warnings.append("No clean exponential window; diode region is only a fallback.")
    else:
        diode_region = (max(0.0, vmin+0.35*span), vmin+0.7*span)
        warnings.append("Too few positive diode-current points for n/J0 region detection.")

    ridx = np.flatnonzero((v > 0) & (j > 0))
    if len(ridx) >= 6:
        # High-current regression; avoid demanding a fixed voltage threshold.
        start = max(0, int(0.68 * len(ridx)))
        if len(ridx) - start < 6:
            start = len(ridx) - 6
        rs_region = (float(v[ridx[start]]), float(v[ridx[-1]]))
    else:
        rs_region = (max(0.0, vmin + 0.72 * span), vmin + 0.95*span)
        warnings.append("Too few high-current points to estimate Rs reliably.")

    # Default final fitting range retains EVERY point.  Only a clearly
    # instrument-like current plateau is optionally excluded at the top.
    fit_lo, fit_hi = vmin, vmax
    if len(ridx) >= 12:
        last = ridx[-5:]
        dv = float(v[last[-1]] - v[last[0]])
        dj = float(np.ptp(j[last]))
        if dv > max(0.04, 0.035*span) and dj < 0.002*max(abs(j[last[-1]]), 1e-20):
            fit_hi = float(v[ridx[-6]])
            # Do not use an excluded compliance plateau for the Rs start.
            rs_region = (min(rs_region[0], fit_hi), fit_hi)
            warnings.append("Possible current-compliance plateau: proposed fit and Rs windows stop before it; verify manually.")

    clamp = lambda p: (float(np.clip(min(p), vmin, vmax)),
                       float(np.clip(max(p), vmin, vmax)))
    return ({"rsh": clamp(rsh_region), "diode": clamp(diode_region),
             "rs": clamp(rs_region), "fit": clamp((fit_lo, fit_hi))}, warnings)


# =============================================================
# FITTING, FREE/FIXED PARAMETERS, AND IDENTIFIABILITY
# =============================================================

def choose_j_scale(J):
    a = np.abs(np.asarray(J))
    a = a[np.isfinite(a) & (a > 0)]
    return max(1e-15, float(np.percentile(a, 15))) if len(a) else 1e-12


def make_residual(prediction, measured, objective, j_scale,
                  sigma_abs=0.0, sigma_rel=0.0):
    if objective == "Linear J-V":
        scale = max(1e-15, float(np.percentile(np.abs(measured), 90)))
        return (prediction - measured) / scale
    if objective == "Log J-V (positive J only)":
        valid = measured > 0
        if np.sum(valid) < 3:
            raise ValueError("Log objective needs at least three positive-current points")
        floor = 1e-30
        p = prediction[valid]
        m = measured[valid]
        # Extra gradient-producing penalty for an unphysical negative prediction.
        return np.log10(np.maximum(p, floor)) - np.log10(np.maximum(m, floor)) \
            + np.maximum(-p, 0) / max(j_scale, 1e-15)
    if objective == "Noise-aware":
        sigma = np.hypot(sigma_abs, sigma_rel * np.abs(measured))
        return (prediction - measured) / np.maximum(sigma, 1e-30)
    return np.arcsinh(prediction / j_scale) - np.arcsinh(measured / j_scale)


def _bounds_for_key(key):
    lo, hi = PARAM_BOUNDS[key]
    return (np.log10(lo), np.log10(hi)) if key in LOG_KEYS else (lo, hi)


def _encode(key, value):
    return float(np.log10(value)) if key in LOG_KEYS else float(value)


def _decode(key, value):
    return float(10.0 ** value) if key in LOG_KEYS else float(value)


def _fit_guess(estimates, two):
    j0 = estimates.J0 if np.isfinite(estimates.J0) and estimates.J0 > 0 else 1e-10
    n = estimates.n if np.isfinite(estimates.n) and estimates.n > 0 else 1.5
    rs = estimates.rs_area if np.isfinite(estimates.rs_area) and estimates.rs_area > 0 else 1.0
    rsh = estimates.rsh_area if np.isfinite(estimates.rsh_area) and estimates.rsh_area > 0 else 1e6
    if not two:
        return dict(J0=j0, n=n, rs=rs, rsh=rsh)
    return dict(J01=0.7*j0, n1=min(n, 2.0), J02=0.3*j0,
                n2=max(1.8, min(n+0.7, 5.0)), rs=rs, rsh=rsh)


def _quality(J, pred, fit_residual):
    raw = J - pred
    rmse = float(np.sqrt(np.mean(raw**2)))
    denom = float(np.sum((J-np.mean(J))**2))
    r2 = float(1 - np.sum(raw**2)/denom) if denom > 0 else np.nan
    rmse_obj = float(np.sqrt(np.mean(np.asarray(fit_residual)**2)))
    good = (J > 0) & (pred > 0)
    log_rmse = (float(np.sqrt(np.mean((np.log10(J[good]) -
                                     np.log10(pred[good]))**2)))
                if np.sum(good) >= 3 else np.nan)
    return raw, rmse, r2, rmse_obj, log_rmse


def _fit_core(V, J, T, estimates, objective, n_starts, *, two=False,
              fixed=None, constrain_two=False, sigma_abs=0.0,
              sigma_rel=0.02, check_sensitivity=True):
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    names = TWO_KEYS if two else SINGLE_KEYS
    fixed = dict(fixed or {})
    if two and constrain_two:
        if "n1" in fixed or "n2" in fixed:
            raise ValueError("Choose either fixed two-diode idealities or individual n1/n2 locks")
        fixed.update(n1=1.0, n2=2.0)
    for k, val in fixed.items():
        if k not in names:
            raise ValueError(f"Invalid fixed parameter: {k}")
        if not PARAM_BOUNDS[k][0] <= val <= PARAM_BOUNDS[k][1]:
            raise ValueError(f"Fixed {k}={val:g} is outside bounds {PARAM_BOUNDS[k]}")

    guess = _fit_guess(estimates, two)
    guess.update(fixed)
    for k in names:
        lo, hi = PARAM_BOUNDS[k]
        guess[k] = float(np.clip(guess[k], lo*1.00001, hi/1.00001)) if k not in fixed else fixed[k]

    free = [k for k in names if k not in fixed]
    lower = np.array([_bounds_for_key(k)[0] for k in free], dtype=float)
    upper = np.array([_bounds_for_key(k)[1] for k in free], dtype=float)
    x0 = np.array([_encode(k, guess[k]) for k in free], dtype=float)
    x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)
    j_scale = choose_j_scale(J)
    n_residual = int(np.sum(J > 0)) if objective == "Log J-V (positive J only)" else len(J)
    if n_residual <= len(free):
        raise ValueError("Too few effective data points for the number of FREE fitting parameters")

    def unpack(x):
        p = dict(fixed)
        p.update({k: _decode(k, v) for k, v in zip(free, x)})
        return p

    def forward(p):
        if two:
            return solve_j_two(V, p["J01"], p["n1"], p["J02"],
                               p["n2"], p["rs"], p["rsh"], T)
        return solve_j_single(V, p["J0"], p["n"], p["rs"], p["rsh"], T)

    def residual(x):
        p = unpack(x)
        try:
            prediction = forward(p)
            return make_residual(prediction, J, objective, j_scale, sigma_abs, sigma_rel)
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(n_residual, 1e12, dtype=float)

    rng = np.random.default_rng(54321 if two else 12345)
    starts = [x0]
    if free:
        for i in range(max(0, n_starts-1)):
            if (i+1) % 6 == 0:
                start = rng.uniform(lower+1e-7, upper-1e-7)
            else:
                sigma = np.array([1.3 if k in LOG_KEYS else 0.35 for k in free])
                start = x0 + rng.normal(0, sigma)
                start = np.clip(start, lower+1e-7, upper-1e-7)
            starts.append(start)

    best = None
    if free:
        for start in starts:
            try:
                result = least_squares(
                    residual, start, bounds=(lower, upper),
                    loss="soft_l1", f_scale=1.0,
                    x_scale="jac", max_nfev=2500 if two else 1800,
                    ftol=1e-9, xtol=1e-10, gtol=1e-9)
                if np.isfinite(result.cost) and (
                    best is None or (result.cost, not result.success) <
                    (best.cost, not best.success)
                ):
                    best = result
            except (ValueError, FloatingPointError):
                continue
        if best is None:
            raise RuntimeError("All nonlinear fitting attempts failed")
        x_best = best.x
        success = bool(best.success)
        cost = float(best.cost)
    else:
        x_best = np.empty(0)
        rr = residual(x_best)
        cost = float(np.sum(np.sqrt(1+rr**2)-1))
        success = True

    params = unpack(x_best)
    prediction = forward(params)
    closure = verify_physical_equation(V, prediction, params, T, two=two)
    if not np.isfinite(closure) or closure > 1e-6:
        raise RuntimeError(f"Numerical diode equation was not satisfied (error={closure:.2e})")
    fitted_res = make_residual(prediction, J, objective, j_scale, sigma_abs, sigma_rel)
    raw, rmse, r2, rmse_obj, log_rmse = _quality(J, prediction, fitted_res)

    warnings = measurement_warnings(V)
    if not success:
        warnings.append("Optimizer did not meet its convergence tolerance.")
    span = upper - lower
    near = [k for k, x, l, s, h in zip(free, x_best, lower, span, upper)
            if ((x-l)/s < 0.01 or (h-x)/s < 0.01)]
    if near:
        warnings.append("Parameter(s) at/near search bounds: " + ", ".join(near))
    if objective == "Log J-V (positive J only)":
        warnings.append("Log objective EXCLUDES negative/zero current; Rsh may be weakly constrained.")
    if len(V) < max(16, 4 * len(free)):
        warnings.append("Few data points relative to fitted parameters.")
    if np.max(V) <= 0 or np.sum((V > 0) & (J > 0)) < 5:
        warnings.append("Insufficient positive forward-current data for reliable diode extraction.")
    if np.min(V) > 0:
        warnings.append("No zero/reverse-bias data: independent Rsh identification may be difficult.")

    if not two:
        voltage_drop = np.max(np.abs(prediction * params["rs"]))
        if voltage_drop < 0.15 * params["n"] * thermal_voltage(T):
            warnings.append("Series-voltage drop is small: Rs may be weakly constrained.")
    else:
        u = V - prediction * params["rs"]
        j1 = params["J01"] * np.expm1(np.minimum(u/(params["n1"]*thermal_voltage(T)), 600))
        j2 = params["J02"] * np.expm1(np.minimum(u/(params["n2"]*thermal_voltage(T)), 600))
        forward_mask = (u > 0) & (prediction > 0)
        if np.sum(forward_mask) >= 3:
            frac = np.abs(j1[forward_mask]) / np.maximum(
                np.abs(j1[forward_mask]) + np.abs(j2[forward_mask]), 1e-30)
            if np.all(frac > 0.98) or np.all(frac < 0.02):
                warnings.append("One diode dominates every forward point; the other diode is unidentifiable.")
        if abs(params["n1"] - params["n2"]) < 0.12:
            warnings.append("n1 and n2 are nearly equal: J01 and J02 cannot be separated reliably.")

    weak = []
    if free and len(free) > 1:
        # Local Jacobian is an inexpensive first identifiability check.
        jac = best.jac
        colnorm = np.linalg.norm(jac, axis=0)
        if np.any(colnorm < 1e-9 * max(np.max(colnorm), 1e-15)):
            warnings.append("One or more fitted parameters have near-zero local sensitivity.")
        if np.all(colnorm > 0):
            scaled = jac / colnorm
            sv = np.linalg.svd(scaled, compute_uv=False)
            if sv[-1] < 1e-5 * sv[0]:
                warnings.append("Strong parameter correlation: values may not be unique.")

    if check_sensitivity and free:
        # Heuristic (NOT a confidence interval): move one parameter by
        # ~factor 3 for log parameters or 0.3 for ideality, reoptimize others.
        # If cost hardly changes, its individual value is not well resolved.
        for index, key in enumerate(free):
            shift = 0.5 if key in LOG_KEYS else 0.3
            trial_costs = []
            for sign in (-1, 1):
                pinned = x_best[index] + sign * shift
                if not lower[index] + 1e-6 < pinned < upper[index] - 1e-6:
                    continue
                remaining = [i for i in range(len(free)) if i != index]
                def profile_residual(z):
                    full = x_best.copy()
                    full[index] = pinned
                    full[remaining] = z
                    return residual(full)
                try:
                    if remaining:
                        profile = least_squares(
                            profile_residual, x_best[remaining],
                            bounds=(lower[remaining], upper[remaining]),
                            loss="soft_l1", f_scale=1.0,
                            max_nfev=300, ftol=3e-6, xtol=3e-7, gtol=3e-6)
                        trial_costs.append(profile.cost)
                    else:
                        rr = profile_residual(np.empty(0))
                        trial_costs.append(np.sum(np.sqrt(1+rr**2)-1))
                except (ValueError, FloatingPointError):
                    continue
            if trial_costs and min(trial_costs)-cost < max(0.03*cost, 1e-9):
                weak.append(key)
        if weak:
            warnings.append("Weakly constrained (profile heuristic): " + ", ".join(weak))

    # If no diode locks were used, call the lower-n contribution diode 1.
    # When diode parameters are FIXED, preserving their labels is essential.
    if two and not any(k in fixed for k in ("J01", "n1", "J02", "n2")):
        if params["n1"] > params["n2"]:
            params["J01"], params["J02"] = params["J02"], params["J01"]
            params["n1"], params["n2"] = params["n2"], params["n1"]
            def swap_labels(xs):
                return [{"J01":"J02", "J02":"J01", "n1":"n2", "n2":"n1"}.get(z, z)
                        for z in xs]
            near = swap_labels(near)
            weak = swap_labels(weak)
            # Diagnostic text must refer to the displayed (sorted) labels.
            for i, message in enumerate(warnings):
                for src, tmp in (("J01", "@DIODEA@"), ("J02", "@DIODEB@"),
                                 ("n1", "@IDEALA@"), ("n2", "@IDEALB@")):
                    message = message.replace(src, tmp)
                for tmp, dest in (("@DIODEA@", "J02"), ("@DIODEB@", "J01"),
                                  ("@IDEALA@", "n2"), ("@IDEALB@", "n1")):
                    message = message.replace(tmp, dest)
                warnings[i] = message

    return dict(
        model="Two diode" if two else "Single diode",
        **params, prediction=prediction, raw_residual=raw,
        RMSE=rmse, R2=r2, Fit_RMSE=rmse_obj, Log_RMSE=log_rmse,
        bound_hits=near, weak_parameters=weak, warnings=warnings,
        success=success, J_scale=j_scale, cost=cost,
        closure=closure, fitted_count=len(V), fixed=fixed.copy(),
    )


