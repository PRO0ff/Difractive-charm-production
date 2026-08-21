"""
Diffractive c cbar production in pp at sqrt(s) = 13.6 TeV
Light-cone dipole approach, leading-twist production mechanism.
Kopeliovich, Potashnikova, Schmidt, Tarasov, PRD 76, 034019 (2007).

Verified pieces (see Section 4):
    T_qp = (1/9) Delta^2 + 2 Gamma^2
    T_Gp = (39/128) Delta'^2 + (27/128) Omega'^2
Both color traces were re-derived and checked numerically.
"""

import numpy as np
from scipy.special import kn

# ----------------------------------------------------------------------
# Units (natural units, hbar = c = 1)
# ----------------------------------------------------------------------
HBARC_GEV_FM = 0.1973269804                  # GeV*fm
FM_TO_GEVINV = 1.0 / HBARC_GEV_FM            # 1 fm  = 5.0677 GeV^-1
GEV2_TO_MB   = 0.389379                      # 1 GeV^-2 = 0.3894 mb
MB_TO_GEV2   = 1.0 / GEV2_TO_MB              # 1 mb   = 2.5682 GeV^-2
MB_TO_UB     = 1.0e3

# ----------------------------------------------------------------------
# Physical parameters
# ----------------------------------------------------------------------
MC       = 1.5                               # charm mass [GeV]
MQ_LIGHT = 0.2                               # effective light-quark mass ~ Lambda_QCD [GeV]
NF       = 3                                 # active flavors for charm
SQRT_S   = 13600.0                           # pp cms energy [GeV]
S        = SQRT_S**2                          # [GeV^2]
MU2      = 4.0 * MC**2                        # factorization scale^2 [GeV^2]

# GBW dipole model (Golec-Biernat & Wuesthoff)
SIGMA0   = 23.03 * MB_TO_GEV2                 # saturation cross section [GeV^-2]
X0       = 3.04e-4
LAMBDA   = 0.288
R0_NORM  = 0.4 * FM_TO_GEVINV                 # 0.4 fm in GeV^-1

# Survival probability inputs at 13.6 TeV
SIG_TOT  = 115.0 * MB_TO_GEV2                 # [GeV^-2]
B_EL     = 21.0                               # [GeV^-2]
B_SD     = 13.0                               # [GeV^-2]

ALPHA_S_FROZEN = 0.4                          # IR freeze-out value (paper: alpha_s <= 0.4)
LAMBDA_QCD     = 0.2                           # [GeV]

# ----------------------------------------------------------------------
# QCD coupling and dipole cross section
# ----------------------------------------------------------------------
def alpha_s(scale_gev):
    """One-loop running coupling, frozen in the infrared at ALPHA_S_FROZEN."""
    scale = max(scale_gev, 1e-6)
    b0 = 11.0 - 2.0 * NF / 3.0                # = 9 for nf=3
    running = 4.0 * np.pi / (b0 * np.log(scale**2 / LAMBDA_QCD**2)) \
              if scale > LAMBDA_QCD * np.e**0.5 else np.inf
    return min(ALPHA_S_FROZEN, running) if np.isfinite(running) else ALPHA_S_FROZEN

def R0(xtilde):
    """GBW saturation radius [GeV^-1]."""
    return R0_NORM * (xtilde / X0) ** (LAMBDA / 2.0)

def gbw_sigma(r_gevinv, xtilde):
    """Universal dipole cross section sigma(r, xtilde) [GeV^-2]. r in GeV^-1."""
    r2 = r_gevinv * r_gevinv
    return SIGMA0 * (1.0 - np.exp(-r2 / R0(xtilde) ** 2))

# ----------------------------------------------------------------------
# Geometry helper: |a*rho_vec + b*s_vec| with s along x-axis, rho at angle phi
# ----------------------------------------------------------------------
def _mag(a, rho, phi, b, s):
    x = a * rho * np.cos(phi) + b * s
    y = a * rho * np.sin(phi)
    return np.hypot(x, y)

# ----------------------------------------------------------------------
# Partonic production-mechanism integrand (VERIFIED core, Section 4)
# ----------------------------------------------------------------------
def _wavefunctions(rho, s, alpha, beta):
    """Return (L, T_quark, T_gluon) shape factors (couplings applied outside)."""
    x = np.sqrt(s**2 + (1.0 - alpha) * rho**2 / (beta * (1.0 - beta)))
    # longitudinal (quark) and gluon-longitudinal
    L_q = 16.0 * (1.0 - alpha)**2 * (MC**2 / x**2) * kn(1, MC * x)**2
    L_g = 16.0 * (MC**2 / x**2) * kn(1, MC * x)**2
    # transverse radial factor V = m_Q (K1(mQ s) - (s/x) K1(mQ x))
    V = MC * (kn(1, MC * s) - (s / x) * kn(1, MC * x))
    spin_beta = beta**2 + (1.0 - beta)**2
    T_q = (2.0 * (1.0 + (1.0 - alpha)**2) / rho**2) * spin_beta * V**2
    T_g = (4.0 * (1.0 + (1.0 - alpha)**4 + alpha**4)
           / (alpha**2 * (1.0 - alpha)**2 * rho**2)) * spin_beta * V**2
    return L_q, T_q, L_g, T_g

def _trace_terms(rho, phi, s, alpha, beta, xtilde):
    """Return (T_qp, T_Gp) color traces of Section 4."""
    sig = lambda a, b: gbw_sigma(_mag(a, rho, phi, b, s), xtilde)
    # --- quark: Delta, Lambda, Gamma ---
    Delta = ( sig(1.0,  beta)      - sig(1.0, -(1-beta))
            + sig(1-alpha, beta)   - sig(1-alpha, -(1-beta)) )
    s_s      = gbw_sigma(s, xtilde)
    s_1bs    = gbw_sigma((1-beta) * s, xtilde)
    s_bs     = gbw_sigma(beta * s, xtilde)
    s_rho    = gbw_sigma(rho, xtilde)
    s_1arho  = gbw_sigma((1-alpha) * rho, xtilde)
    Lambda = ( s_s - s_1bs - s_bs + s_rho - s_1arho
             + 0.5 * sig(1-alpha, -(1-beta)) + 0.5 * sig(1-alpha, beta)
             - 0.5 * sig(1.0, -(1-beta))     - 0.5 * sig(1.0, beta) )
    Gamma = (5.0/24.0) * Delta + (7.0/12.0) * s_s + 0.75 * Lambda
    T_qp = (1.0/9.0) * Delta**2 + 2.0 * Gamma**2
    # --- gluon: Delta', Omega' ---
    Dp = ( sig(1-alpha, -(1-beta)) + sig(1.0, -(1-beta))
         - sig(1-alpha, beta)      - sig(1.0, beta) )
    Op = ( 2.0 * s_1arho - sig(1-alpha, beta) - sig(1-alpha, -(1-beta))
         - 2.0 * s_rho    + sig(1.0, beta)    + sig(1.0, -(1-beta))
         - 2.0 * s_bs     - 2.0 * s_1bs       + (32.0/9.0) * s_s )
    T_Gp = (39.0/128.0) * Dp**2 + (27.0/128.0) * Op**2
    return T_qp, T_Gp

# Fixed integration grids. rho and s are log-spaced (scales span 1/m_Q .. 1/Lambda);
# phi is the relative azimuth between rho and s. The common overall azimuth gives 2*pi.
_RHO = np.logspace(np.log10(1e-2), np.log10(8.0), 80)     # [GeV^-1]
_S   = np.logspace(np.log10(3e-3), np.log10(4.0), 80)     # [GeV^-1]
_PHI = np.linspace(0.0, 2.0 * np.pi, 48)
_ALS = np.array([alpha_s(1.0 / r)  for r  in _RHO])        # soft coupling, depends on rho
_ASS = np.array([alpha_s(1.0 / ss) for ss in _S])          # hard coupling, depends on s

def dsigma_dt_dadb(alpha, beta, xtilde, channel):
    """
    d sigma_Pr / dt' dalpha dbeta at t'=0 [GeV^-4] for a single parton (vectorized).
    channel = 'quark' or 'gluon'.  Uses a fixed log-grid in (rho, s) and linear in phi.
    """
    RHO = _RHO[:, None, None]                  # (Nrho,1,1)
    PHI = _PHI[None, :, None]                  # (1,Nphi,1)
    SS  = _S[None, None, :]                     # (1,1,Ns)

    L_q, T_q, L_g, T_g = _wavefunctions(RHO, SS, alpha, beta)
    T_qp, T_Gp = _trace_terms(RHO, PHI, SS, alpha, beta, xtilde)
    coupling = (_ALS[:, None, None] * _ASS[None, None, :]) / (2.0 * np.pi)**4

    if channel == 'quark':
        wf_sq, trace = coupling * (L_q + T_q), T_qp
    else:
        wf_sq, trace = coupling * (L_g + T_g), T_Gp

    integrand = wf_sq * trace * RHO * SS        # measure factor rho*s (from d^2rho d^2s)
    # integrate phi, then s, then rho; common azimuth contributes an extra 2*pi
    over_phi = np.trapezoid(integrand, _PHI, axis=1)          # (Nrho, Ns)
    over_s   = np.trapezoid(over_phi, _S,  axis=1)            # (Nrho,)
    val      = np.trapezoid(over_s,  _RHO, axis=0) * 2.0 * np.pi
    prefac = (3.0 if channel == 'quark' else 9.0) / (256.0 * np.pi)
    return prefac * val

# ----------------------------------------------------------------------
# Survival probability (Eq. 444)
# ----------------------------------------------------------------------
def survival_K():
    term1 = (1.0 / np.pi) * SIG_TOT / (B_SD + 2.0 * B_EL)
    term2 = (1.0 / (4.0 * np.pi)**2) * SIG_TOT**2 / (B_EL * (B_SD + B_EL))
    return 1.0 - term1 + term2

# ----------------------------------------------------------------------
# Projectile PDFs: use lhapdf if available, else a toy parametrization
# ----------------------------------------------------------------------
try:
    import lhapdf
    _PDF = lhapdf.mkPDF("CT18NLO", 0)
    def pdf_densities(xp):
        """Return (sum_q (q+qbar), g) number densities at scale MU2."""
        quarks = sum(_PDF.xfxQ2(fl, xp, MU2) + _PDF.xfxQ2(-fl, xp, MU2)
                     for fl in (1, 2, 3)) / xp
        glue = _PDF.xfxQ2(21, xp, MU2) / xp
        return quarks, glue
except Exception:
    def pdf_densities(xp):
        """Toy valence+sea+gluon densities (illustrative; swap in a real PDF set)."""
        if xp <= 0.0 or xp >= 1.0:
            return 0.0, 0.0
        uv  = 2.0 * xp**(-0.5) * (1.0 - xp)**3
        dv  = 1.0 * xp**(-0.5) * (1.0 - xp)**4
        sea = 0.4 * xp**(-1.0) * (1.0 - xp)**7
        glue = 3.0 * xp**(-1.0) * (1.0 - xp)**5
        quarks = uv + dv + 2.0 * sea           # sum over light flavors of (q + qbar)
        return quarks, glue

# ----------------------------------------------------------------------
# Hadron-level assembly (model-dependent, Section 5)
# ----------------------------------------------------------------------
def sigma_pp_charm(xF_min=0.85, n_alpha=6, n_beta=6, n_xp=6):
    """
    Diffractive cross section sigma(pp -> ccbar X p) [micro-barn],
    integrated over x_F > xF_min. Coarse grid; raise n_* for precision.

    Additive model (Eq. 452):
      dsigma_pp = K/B_sd * (1/3) [ (q+qbar) dsigma_qp + (81/16) g dsigma_Gp ]
                  integrated over alpha, beta, x_Pomeron.
    """
    K = survival_K()
    xp_max = 1.0 - xF_min
    a_grid  = np.linspace(0.05, 0.95, n_alpha)
    b_grid  = np.linspace(0.05, 0.95, n_beta)
    xP_grid = np.linspace(1e-3, xp_max, n_xp)
    da, db, dxP = (g[1] - g[0] for g in (a_grid, b_grid, xP_grid))

    x1_min = 4.0 * MC**2 / ((1.0 - xF_min) * S)   # kinematic threshold
    total = 0.0
    for xP in xP_grid:
        for al in a_grid:
            xp = xP / al                           # active-parton momentum fraction
            if xp >= 1.0:
                continue
            xtilde = 4.0 * MC**2 / (xp * S)         # dipole Bjorken-x
            q_sum, glue = pdf_densities(xp)
            for be in b_grid:
                if al * be < x1_min:                # charm momentum-fraction bound
                    continue
                dq = dsigma_dt_dadb(al, be, xtilde, 'quark')
                dg = dsigma_dt_dadb(al, be, xtilde, 'gluon')
                folded = (q_sum * dq + (81.0/16.0) * glue * dg) / 3.0
                total += folded * da * db * dxP
    sigma_gev2 = K * total / B_SD
    return sigma_gev2 * GEV2_TO_MB * MB_TO_UB

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(f"Survival probability K(13.6 TeV) = {survival_K():.3f}")

    # --- Verified partonic differentials at a representative point (fast) ---
    alpha, beta = 0.5, 0.5
    xtilde = 4.0 * MC**2 / (0.05 * S)          # xp ~ 0.05 representative
    dq = dsigma_dt_dadb(alpha, beta, xtilde, 'quark')
    dg = dsigma_dt_dadb(alpha, beta, xtilde, 'gluon')
    print(f"[alpha=beta=0.5, xtilde={xtilde:.2e}]")
    print(f"  dsigma_qp/dt'dalpha dbeta |_t'=0 = {dq:.4e} GeV^-4")
    print(f"  dsigma_Gp/dt'dalpha dbeta |_t'=0 = {dg:.4e} GeV^-4")

    # --- Hadron-level assembly: UNVALIDATED SCAFFOLD (see caveat) ---
    # Uncomment to run; the current absolute normalization is NOT trustworthy.
    sig = sigma_pp_charm(xF_min=0.85, n_alpha=6, n_beta=6, n_xp=6)
    print(f"[scaffold] sigma(pp->ccbar X p), x_F>0.85 ~ {sig:.2f} ub  (do NOT quote)")


"""Append these functions below the model in pasted-text.txt, before its Main block.

They return d sigma / d pT^2.  The variable pT is the transverse momentum
conjugate to the heavy-quark separation ``s`` (the paper's kappa in Eqs.
(44) and (50)); it is the pT plotted in Figs. 10-12.

The original code contains only diagonal coordinate-space probabilities, so
it is already integrated over pT.  Recovering the exact spectrum requires
the off-diagonal amplitude Phi(rho, s1) Phi*(rho, s2), which is not present
in that code.  The diffractive function below therefore uses a normalized
Gaussian-mixture Fourier ansatz

    K(pT^2, s) = s^2 exp(-pT^2 s^2),  integral K d pT^2 = 1.

This preserves the original integrated diffractive cross section exactly
(when integrated from pT^2 = 0 to infinity), while supplying a smooth pT
dependence.  It is a phenomenological extension, not a digitization of the
paper's Fig. 10.

The non-diffractive shape uses the inclusive QQbar-g dipole cross section
from Eq. (26) of arXiv:hep-ph/0702106.  Its absolute pp normalization must be
supplied: the original model does not implement the inclusive pp PDF folding
or the full non-diffractive off-diagonal kernel.
"""


def _pt2_array(pT2):
    """Validate pT^2 [GeV^2] and return (flat values, input_was_scalar)."""
    values = np.asarray(pT2, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("pT2 must contain finite, non-negative values in GeV^2")
    return np.atleast_1d(values), values.ndim == 0


def _restore_pt2_shape(values, scalar):
    return float(values[0]) if scalar else values


def dsigma_dt_dadb_dpT2(alpha, beta, xtilde, channel, pT2):
    """Diffractive partonic spectrum d sigma/(dt' d alpha d beta d pT^2).

    Parameters have the same meaning as ``dsigma_dt_dadb``; ``pT2`` is in
    GeV^2.  The result has units GeV^-6 and is vectorized over ``pT2``.

    The Gaussian-mixture kernel is normalized in pT^2, so integrating this
    function over pT^2 reproduces ``dsigma_dt_dadb``.
    """
    if channel not in {"quark", "gluon"}:
        raise ValueError("channel must be 'quark' or 'gluon'")
    pT2_values, scalar = _pt2_array(pT2)

    RHO = _RHO[:, None, None]
    PHI = _PHI[None, :, None]
    SS = _S[None, None, :]

    L_q, T_q, L_g, T_g = _wavefunctions(RHO, SS, alpha, beta)
    T_qp, T_Gp = _trace_terms(RHO, PHI, SS, alpha, beta, xtilde)
    coupling = (_ALS[:, None, None] * _ASS[None, None, :]) / (2.0 * np.pi)**4

    if channel == "quark":
        integrand = coupling * (L_q + T_q) * T_qp * RHO * SS
        prefactor = 3.0 / (256.0 * np.pi)
    else:
        integrand = coupling * (L_g + T_g) * T_Gp * RHO * SS
        prefactor = 9.0 / (256.0 * np.pi)

    # s is conjugate to the relative heavy-quark momentum.  This kernel has
    # units GeV^-2 and integrates to one over pT^2 in [0, infinity).
    kernel = SS[None, :, :, :]**2 * np.exp(
        -pT2_values[:, None, None, None] * SS[None, :, :, :]**2
    )
    weighted = integrand[None, :, :, :] * kernel
    over_phi = np.trapezoid(weighted, _PHI, axis=2)
    over_s = np.trapezoid(over_phi, _S, axis=2)
    result = prefactor * np.trapezoid(over_s, _RHO, axis=1) * 2.0 * np.pi
    return _restore_pt2_shape(result, scalar)


def dsigma_pp_charm_diff_dpT2(pT2, xF_min=0.85, n_alpha=6, n_beta=6, n_xp=6):
    """Diffractive pp charm spectrum d sigma/dpT^2 [microbarn / GeV^2].

    This is the pT-differential counterpart of ``sigma_pp_charm``.  The
    inexpensive grids (six points per longitudinal variable) are suitable for
    an exploratory curve; increase all three grid counts only after checking
    numerical convergence.
    """
    if not 0.0 < xF_min < 1.0:
        raise ValueError("xF_min must lie strictly between 0 and 1")
    if min(n_alpha, n_beta, n_xp) < 2:
        raise ValueError("n_alpha, n_beta, and n_xp must each be at least 2")

    pT2_values, scalar = _pt2_array(pT2)
    xP_max = 1.0 - xF_min
    a_grid = np.linspace(0.05, 0.95, n_alpha)
    b_grid = np.linspace(0.05, 0.95, n_beta)
    xP_grid = np.linspace(1e-3, xP_max, n_xp)
    da, db, dxP = (grid[1] - grid[0] for grid in (a_grid, b_grid, xP_grid))

    x1_min = 4.0 * MC**2 / ((1.0 - xF_min) * S)
    total = np.zeros_like(pT2_values)
    for xP in xP_grid:
        for alpha in a_grid:
            xp = xP / alpha
            if xp >= 1.0:
                continue
            xtilde = 4.0 * MC**2 / (xp * S)
            q_sum, glue = pdf_densities(xp)
            for beta in b_grid:
                if alpha * beta < x1_min:
                    continue
                quark = dsigma_dt_dadb_dpT2(
                    alpha, beta, xtilde, "quark", pT2_values
                )
                gluon = dsigma_dt_dadb_dpT2(
                    alpha, beta, xtilde, "gluon", pT2_values
                )
                folded = (q_sum * quark + (81.0 / 16.0) * glue * gluon) / 3.0
                total += folded * da * db * dxP

    result = survival_K() * total / B_SD * GEV2_TO_MB * MB_TO_UB
    return _restore_pt2_shape(result, scalar)


def nondiff_charm_pt_shape(pT2, x_target=1.0e-2, n_beta=80):
    """Unit-normalized inclusive charm shape dP/dpT^2 [GeV^-2].

    This implements the radial QQbar-g dipole weight of Eq. (26):
        Sigma_2(s,beta) = 9/8 [sigma(beta*s) + sigma((1-beta)*s)]
                           - 1/8 sigma(s).
    Its integral over pT^2 is one.  It intentionally has no pp normalization.
    """
    if not 0.0 < x_target < 1.0:
        raise ValueError("x_target must lie strictly between 0 and 1")
    if n_beta < 2:
        raise ValueError("n_beta must be at least 2")

    pT2_values, scalar = _pt2_array(pT2)
    beta = np.linspace(1.0e-3, 1.0 - 1.0e-3, n_beta)[:, None]
    s_coord = _S[None, :]
    xtilde = 4.0 * MC**2 / (x_target * S)

    k0 = kn(0, MC * s_coord)
    k1 = kn(1, MC * s_coord)
    splitting = beta**2 + (1.0 - beta)**2
    psi_sq = MC**2 * (k0**2 + splitting * k1**2)
    sigma_qqg = (
        9.0 / 8.0
        * (gbw_sigma(beta * s_coord, xtilde)
           + gbw_sigma((1.0 - beta) * s_coord, xtilde))
        - 1.0 / 8.0 * gbw_sigma(s_coord, xtilde)
    )
    radial_weight = psi_sq * sigma_qqg * s_coord

    normalization = np.trapezoid(
        np.trapezoid(radial_weight, _S, axis=1), beta[:, 0], axis=0
    )
    kernel = s_coord[None, :, :]**2 * np.exp(
        -pT2_values[:, None, None] * s_coord[None, :, :]**2
    )
    differential = np.trapezoid(
        np.trapezoid(radial_weight[None, :, :] * kernel, _S, axis=2),
        beta[:, 0],
        axis=1,
    )
    return _restore_pt2_shape(differential / normalization, scalar)


def dsigma_pp_charm_nondiff_dpT2(pT2, sigma_nondiff_ub, x_target=1.0e-2,
                                  n_beta=80):
    """Non-diffractive pp charm spectrum d sigma/dpT^2 [microbarn / GeV^2].

    ``sigma_nondiff_ub`` is the inclusive pp charm cross section in the same
    acceptance used for the comparison.  Supply it from a dedicated inclusive
    calculation or measurement; it cannot be obtained from the diffractive
    PDF folding in the original script.
    """
    if not np.isfinite(sigma_nondiff_ub) or sigma_nondiff_ub <= 0.0:
        raise ValueError("sigma_nondiff_ub must be a positive finite cross section")
    return sigma_nondiff_ub * nondiff_charm_pt_shape(
        pT2, x_target=x_target, n_beta=n_beta
    )


def pp_charm_pt_spectra(pT, sigma_nondiff_ub, xF_min=0.85,
                        n_alpha=6, n_beta=6, n_xp=6, x_target=1.0e-2):
    """Return diffractive and non-diffractive charm spectra on a pT grid.

    Parameters
    ----------
    pT : float or array-like
        Charm-quark transverse momentum in GeV.
    sigma_nondiff_ub : float
        Inclusive non-diffractive charm cross section [microbarn] in the
        matching fiducial region.

    Returns
    -------
    dict
        ``pT``, ``pT2``, ``diffractive`` and ``nondiffractive``.  The two
        spectra are d sigma/dpT^2 in microbarn/GeV^2.  To plot d sigma/dpT,
        multiply either spectrum by ``2 * pT``.
    """
    pT_values = np.asarray(pT, dtype=float)
    if np.any(~np.isfinite(pT_values)) or np.any(pT_values < 0.0):
        raise ValueError("pT must contain finite, non-negative values in GeV")
    pT2 = pT_values**2
    return {
        "pT": pT_values,
        "pT2": pT2,
        "diffractive": dsigma_pp_charm_diff_dpT2(
            pT2, xF_min=xF_min, n_alpha=n_alpha, n_beta=n_beta, n_xp=n_xp
        ),
        "nondiffractive": dsigma_pp_charm_nondiff_dpT2(
            pT2, sigma_nondiff_ub=sigma_nondiff_ub, x_target=x_target
        ),
    }


# Legacy pT-spectrum example (left disabled to avoid an expensive run):
# pT = np.linspace(0.0, 6.0, 13)
# spectra = pp_charm_pt_spectra(
#     pT, sigma_nondiff_ub=1.0e4, n_alpha=4, n_beta=4, n_xp=4
# )
# print(spectra["diffractive"])

# ======================================================================
# Reusable diffractive heavy-flavour appendix (production mechanism)
# ======================================================================
"""Reusable appendix for diffractive heavy-flavour production.

This module implements the leading-twist *production mechanism* in
Kopeliovich, Potashnikova, Schmidt and Tarasov, Phys. Rev. D 76, 034019
(2007), arXiv:hep-ph/0702106:

* the quark and gluon forward cross sections, Eqs. (47) and (54);
* the incoherent pp parton replacement, Eq. (72);
* the eikonal rapidity-gap survival factor, Eq. (70);
* an amplitude-level relative-heavy-quark pT spectrum obtained before the
  Parseval integration over kappa in Eqs. (44) and (50).

The pT spectrum is evaluated by deterministic Monte Carlo integration of the
off-diagonal coordinate-space kernel.  It is not the Gaussian pT ansatz used
in the earlier extension file.  The returned Monte Carlo uncertainty is part
of the result and must be checked before quoting a number.

Physics scope
-------------
The supplied ``HeavyFlavorKernel`` supports c, b, and t production and is
directly reusable at different energies, x_F cuts, dipole parameter sets, and
PDF sets.  Other diffractive processes can reuse ``DiffractivePairCalculator``
by providing a kernel with the same two methods as ``HeavyFlavorKernel``:
``wave_overlap`` and ``trace_cross``.  They must be derived for that process;
heavy-flavour formulae must not be used for Drell-Yan, vector bosons, etc.

Absolute normalisation
----------------------
Use an LHAPDF set for a physics prediction.  ``ToyPDF`` is available only for
testing the numerical pipeline and deliberately requires an explicit opt-in.
The gap-survival inputs are model dependent, so they are explicit arguments.

Dependencies: numpy, scipy; lhapdf is optional but strongly recommended.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

try:
    from scipy.special import j0, k0, k1
except ImportError as exc:  # pragma: no cover - dependency diagnostic
    raise ImportError(
        "This appendix needs scipy. Install it with: python -m pip install scipy"
    ) from exc


# Natural-unit conversions.
HBARC_GEV_FM = 0.1973269804
FM_TO_GEVINV = 1.0 / HBARC_GEV_FM
GEV2_TO_MB = 0.389379
MB_TO_GEV2 = 1.0 / GEV2_TO_MB
MB_TO_UB = 1.0e3


@dataclass(frozen=True)
class HeavyFlavor:
    """Heavy flavour and its perturbative factorisation scale."""

    name: str
    mass: float
    nf: int

    @property
    def mu2(self) -> float:
        return 4.0 * self.mass**2


CHARM = HeavyFlavor("charm", mass=1.5, nf=3)
BEAUTY = HeavyFlavor("beauty", mass=4.7, nf=4)
TOP = HeavyFlavor("top", mass=175.0, nf=5)


@dataclass(frozen=True)
class GBWParameters:
    """GBW dipole parameters used in Eq. (71)."""

    sigma0_mb: float = 23.03
    x0: float = 3.04e-4
    lam: float = 0.288
    r0_norm_fm: float = 0.4

    @property
    def sigma0(self) -> float:
        return self.sigma0_mb * MB_TO_GEV2

    @property
    def r0_norm(self) -> float:
        return self.r0_norm_fm * FM_TO_GEVINV

    def r0(self, xtilde: float) -> float:
        if xtilde <= 0.0:
            raise ValueError("xtilde must be positive")
        return self.r0_norm * (xtilde / self.x0) ** (self.lam / 2.0)

    def sigma(self, radius: np.ndarray | float, xtilde: float) -> np.ndarray:
        radius = np.asarray(radius, dtype=float)
        return self.sigma0 * (1.0 - np.exp(-radius**2 / self.r0(xtilde) ** 2))


@dataclass(frozen=True)
class GapSurvival:
    """Inputs to Eq. (70), with cross section and slopes in natural units."""

    sigma_tot_mb: float
    b_el: float
    b_sd: float

    def probability(self) -> float:
        sigma_tot = self.sigma_tot_mb * MB_TO_GEV2
        value = (
            1.0
            - sigma_tot / (np.pi * (self.b_sd + 2.0 * self.b_el))
            + sigma_tot**2 / ((4.0 * np.pi) ** 2 * self.b_el * (self.b_sd + self.b_el))
        )
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"Eq. (70) gives K={value:.3g}; check the supplied survival inputs"
            )
        return value


class PDFProvider(Protocol):
    """Projectile parton densities as number densities, not x times PDFs."""

    def densities(self, x: float, mu2: float) -> tuple[float, float]:
        """Return (sum_q [q + qbar], gluon)."""


class LHAPDFProvider:
    """A real LHAPDF projectile PDF provider, e.g. ``CT18NLO``."""

    def __init__(self, set_name: str = "CT18NLO", member: int = 0, nf: int = 3):
        try:
            import lhapdf
        except ImportError as exc:  # pragma: no cover - runtime setup check
            raise ImportError(
                "LHAPDF is required for a physics prediction. Install LHAPDF and "
                "its PDF data, or use ToyPDF only for a diagnostic run."
            ) from exc
        self._pdf = lhapdf.mkPDF(set_name, member)
        self._flavours = tuple(range(1, nf + 1))

    def densities(self, x: float, mu2: float) -> tuple[float, float]:
        if not 0.0 < x < 1.0:
            return 0.0, 0.0
        quarks = sum(
            self._pdf.xfxQ2(flavour, x, mu2) + self._pdf.xfxQ2(-flavour, x, mu2)
            for flavour in self._flavours
        ) / x
        gluon = self._pdf.xfxQ2(21, x, mu2) / x
        return float(quarks), float(gluon)


#class ToyPDF:
#    """Diagnostic-only PDF. Do not use its normalisation in a result."""
#
#    def densities(self, x: float, mu2: float) -> tuple[float, float]:  # noqa: ARG002
#        if not 0.0 < x < 1.0:
#            return 0.0, 0.0
#        uv = 2.0 * x ** (-0.5) * (1.0 - x) ** 3
#       dv = x ** (-0.5) * (1.0 - x) ** 4
#       sea = 0.4 * x ** (-1.0) * (1.0 - x) ** 7
#        gluon = 3.0 * x ** (-1.0) * (1.0 - x) ** 5
#        return uv + dv + 2.0 * sea, gluon


@dataclass(frozen=True)
class IntegrationSettings:
    """Numerical controls; increase each setting and require stable results."""

    n_rho: int = 56
    n_s: int = 56
    n_phi: int = 40
    n_samples: int = 2**13
    rho_min: float = 1.0e-2
    rho_max: float = 8.0
    s_min: float = 3.0e-3
    s_max: float = 4.0
    seed: int = 20260721

    def __post_init__(self) -> None:
        if min(self.n_rho, self.n_s, self.n_phi, self.n_samples) < 2:
            raise ValueError("all grid/sample counts must be at least two")
        if not (0.0 < self.rho_min < self.rho_max and 0.0 < self.s_min < self.s_max):
            raise ValueError("invalid transverse-coordinate bounds")


class HeavyFlavorKernel:
    """Paper Eqs. (45), (51), (46), and (53), for the production mechanism."""

    def __init__(self, flavor: HeavyFlavor, gbw: GBWParameters, alpha_s_frozen: float = 0.4,
                 lambda_qcd: float = 0.2):
        self.flavor = flavor
        self.gbw = gbw
        self.alpha_s_frozen = alpha_s_frozen
        self.lambda_qcd = lambda_qcd

    def alpha_s(self, scale: np.ndarray | float) -> np.ndarray:
        scale = np.maximum(np.asarray(scale, dtype=float), 1.0e-12)
        b0 = 11.0 - 2.0 * self.flavor.nf / 3.0
        safe_scale = np.maximum(scale, self.lambda_qcd * np.exp(0.5))
        running = 4.0 * np.pi / (b0 * np.log((safe_scale / self.lambda_qcd) ** 2))
        return np.minimum(running, self.alpha_s_frozen)

    @staticmethod
    def _mag(a: float, rho: np.ndarray, phi: np.ndarray, b: float, s: np.ndarray) -> np.ndarray:
        """Magnitude of a*rho_vec + b*s_vec, with rho along the x axis."""
        return np.hypot(a * rho + b * s * np.cos(phi), b * s * np.sin(phi))

    def trace_components(self, rho: np.ndarray, phi: np.ndarray, s: np.ndarray,
                         alpha: float, beta: float, xtilde: float) -> tuple[np.ndarray, ...]:
        """Return (Delta, Gamma, Delta_prime, Omega_prime) for one QQbar size."""
        sigma = lambda a, b: self.gbw.sigma(self._mag(a, rho, phi, b, s), xtilde)
        sigma_s = self.gbw.sigma(s, xtilde)
        sigma_1bs = self.gbw.sigma((1.0 - beta) * s, xtilde)
        sigma_bs = self.gbw.sigma(beta * s, xtilde)
        sigma_rho = self.gbw.sigma(rho, xtilde)
        sigma_1arho = self.gbw.sigma((1.0 - alpha) * rho, xtilde)

        delta = (
            sigma(1.0, beta) - sigma(1.0, -(1.0 - beta))
            + sigma(1.0 - alpha, beta) - sigma(1.0 - alpha, -(1.0 - beta))
        )
        lam = (
            sigma_s - sigma_1bs - sigma_bs + sigma_rho - sigma_1arho
            + 0.5 * sigma(1.0 - alpha, -(1.0 - beta))
            + 0.5 * sigma(1.0 - alpha, beta)
            - 0.5 * sigma(1.0, -(1.0 - beta))
            - 0.5 * sigma(1.0, beta)
        )
        gamma = 5.0 / 24.0 * delta + 7.0 / 12.0 * sigma_s + 0.75 * lam

        delta_prime = (
            sigma(1.0 - alpha, -(1.0 - beta)) + sigma(1.0, -(1.0 - beta))
            - sigma(1.0 - alpha, beta) - sigma(1.0, beta)
        )
        omega_prime = (
            2.0 * sigma_1arho - sigma(1.0 - alpha, beta)
            - sigma(1.0 - alpha, -(1.0 - beta)) - 2.0 * sigma_rho
            + sigma(1.0, beta) + sigma(1.0, -(1.0 - beta)) - 2.0 * sigma_bs
            - 2.0 * sigma_1bs + 32.0 / 9.0 * sigma_s
        )
        return delta, gamma, delta_prime, omega_prime

    def wave_overlap(self, rho: np.ndarray, s1: np.ndarray, s2: np.ndarray,
                     relative_s_angle: np.ndarray, alpha: float, beta: float,
                     channel: str) -> np.ndarray:
        """Spin-averaged Phi(rho,s1) Phi*(rho,s2), Eqs. (45) and (51)."""
        if channel not in {"quark", "gluon"}:
            raise ValueError("channel must be 'quark' or 'gluon'")
        mass = self.flavor.mass
        x1 = np.sqrt(s1**2 + (1.0 - alpha) * rho**2 / (beta * (1.0 - beta)))
        x2 = np.sqrt(s2**2 + (1.0 - alpha) * rho**2 / (beta * (1.0 - beta)))
        coupling = self.alpha_s(1.0 / rho) * np.sqrt(
            self.alpha_s(1.0 / s1) * self.alpha_s(1.0 / s2)
        ) / (2.0 * np.pi) ** 4
        spin_beta = beta**2 + (1.0 - beta)**2
        v1 = mass * (k1(mass * s1) - s1 / x1 * k1(mass * x1))
        v2 = mass * (k1(mass * s2) - s2 / x2 * k1(mass * x2))
        if channel == "quark":
            longitudinal = 16.0 * (1.0 - alpha)**2 * mass**2 / (x1 * x2) * k1(mass * x1) * k1(mass * x2)
            transverse_prefactor = 2.0 * (1.0 + (1.0 - alpha)**2) / rho**2
        else:
            longitudinal = 16.0 * mass**2 / (x1 * x2) * k1(mass * x1) * k1(mass * x2)
            transverse_prefactor = (
                4.0 * (1.0 + (1.0 - alpha)**4 + alpha**4)
                / (alpha**2 * (1.0 - alpha)**2 * rho**2)
            )
        transverse = transverse_prefactor * spin_beta * v1 * v2 * np.cos(relative_s_angle)
        return coupling * (longitudinal + transverse)

    def trace_cross(self, rho: np.ndarray, s1: np.ndarray, phi1: np.ndarray,
                    s2: np.ndarray, phi2: np.ndarray, alpha: float, beta: float,
                    xtilde: float, channel: str) -> np.ndarray:
        """Colour trace Sigma(s1) Sigma^dagger(s2), before s1=s2 is imposed."""
        d1, g1, dp1, op1 = self.trace_components(rho, phi1, s1, alpha, beta, xtilde)
        d2, g2, dp2, op2 = self.trace_components(rho, phi2, s2, alpha, beta, xtilde)
        if channel == "quark":
            return d1 * d2 / 9.0 + 2.0 * g1 * g2
        if channel == "gluon":
            return 39.0 / 128.0 * dp1 * dp2 + 27.0 / 128.0 * op1 * op2
        raise ValueError("channel must be 'quark' or 'gluon'")


class DiffractivePairCalculator:
    """Numerical calculator for forward and pT-differential pair production."""

    def __init__(self, kernel: HeavyFlavorKernel, pdf: PDFProvider,
                 settings: IntegrationSettings = IntegrationSettings()):
        self.kernel = kernel
        self.pdf = pdf
        self.settings = settings
        self._rho = np.geomspace(settings.rho_min, settings.rho_max, settings.n_rho)
        self._s = np.geomspace(settings.s_min, settings.s_max, settings.n_s)
        self._phi = np.linspace(0.0, 2.0 * np.pi, settings.n_phi)

    def partonic_forward(self, alpha: float, beta: float, xtilde: float, channel: str) -> float:
        """d sigma/(dt' d alpha d beta) at t'=0 [GeV^-4], Eqs. (47)/(54)."""
        rho = self._rho[:, None, None]
        phi = self._phi[None, :, None]
        s_coord = self._s[None, None, :]
        overlap = self.kernel.wave_overlap(rho, s_coord, s_coord, 0.0, alpha, beta, channel)
        trace = self.kernel.trace_cross(
            rho, s_coord, phi, s_coord, phi, alpha, beta, xtilde, channel
        )
        integrand = overlap * trace * rho * s_coord
        over_phi = np.trapezoid(integrand, self._phi, axis=1)
        over_s = np.trapezoid(over_phi, self._s, axis=1)
        prefactor = (3.0 if channel == "quark" else 9.0) / (256.0 * np.pi)
        return float(prefactor * 2.0 * np.pi * np.trapezoid(over_s, self._rho, axis=0))

    def _samples(self, seed_offset: int) -> tuple[np.ndarray, ...]:
        """Sample the five-dimensional reduced off-diagonal integral."""
        rng = np.random.default_rng(self.settings.seed + seed_offset)
        u = rng.random((self.settings.n_samples, 5))
        log_rho_min, log_rho_max = np.log(self.settings.rho_min), np.log(self.settings.rho_max)
        log_s_min, log_s_max = np.log(self.settings.s_min), np.log(self.settings.s_max)
        rho = np.exp(log_rho_min + (log_rho_max - log_rho_min) * u[:, 0])
        s1 = np.exp(log_s_min + (log_s_max - log_s_min) * u[:, 1])
        s2 = np.exp(log_s_min + (log_s_max - log_s_min) * u[:, 2])
        phi1 = 2.0 * np.pi * u[:, 3]
        phi2 = 2.0 * np.pi * u[:, 4]
        jacobian = (
            rho**2 * s1**2 * s2**2
            * (log_rho_max - log_rho_min)
            * (log_s_max - log_s_min) ** 2
            * (2.0 * np.pi) ** 2
        )
        return rho, s1, s2, phi1, phi2, jacobian

    def partonic_pt2(self, alpha: float, beta: float, xtilde: float, channel: str,
                      pT2: np.ndarray | float, seed_offset: int = 0,
                      calculate_mc_error: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """d sigma/(dt' d alpha d beta d pT^2) and optional MC error [GeV^-6].

        This is the off-diagonal Fourier form of Eqs. (44)/(50), with the
        azimuth of kappa analytically integrated to J0(kappa*|s1-s2|).
        Increase ``n_samples`` until both value and error stabilise.
        """
        pT2_values = np.atleast_1d(np.asarray(pT2, dtype=float))
        if np.any(~np.isfinite(pT2_values)) or np.any(pT2_values < 0.0):
            raise ValueError("pT2 must be finite and non-negative")
        rho, s1, s2, phi1, phi2, jacobian = self._samples(seed_offset)
        relative_angle = phi1 - phi2
        overlap = self.kernel.wave_overlap(rho, s1, s2, relative_angle, alpha, beta, channel)
        trace = self.kernel.trace_cross(rho, s1, phi1, s2, phi2, alpha, beta, xtilde, channel)
        distance = np.sqrt(np.maximum(s1**2 + s2**2 - 2.0 * s1 * s2 * np.cos(relative_angle), 0.0))
        prefactor = (3.0 if channel == "quark" else 9.0) / (256.0 * np.pi)
        samples = prefactor * jacobian[None, :] * overlap[None, :] * trace[None, :] * j0(
            np.sqrt(pT2_values)[:, None] * distance[None, :]
        )
        mean = np.mean(samples, axis=1)
        error = (
            np.std(samples, axis=1, ddof=1) / np.sqrt(self.settings.n_samples)
            if calculate_mc_error
            else np.zeros_like(mean)
        )
        return mean, error

    def pp_pt2_spectrum(self, pT2: np.ndarray | float, sqrt_s: float, survival: GapSurvival,
                        xF_min: float = 0.85, n_alpha: int = 4, n_beta: int = 4,
                        n_xp: int = 4, calculate_mc_error: bool = True) -> dict[str, np.ndarray | float]:
        """pp -> QQbar X p production spectrum, Eq. (72), in microbarn/GeV^2.

        ``xP = 1-xF`` is integrated from 10^-3 to ``1-xF_min``.  The result is
        production-mechanism-only, as used for the paper's numerical pT plots.
        """
        if sqrt_s <= 0.0 or not 0.0 < xF_min < 1.0:
            raise ValueError("sqrt_s must be positive and xF_min must lie in (0, 1)")
        if min(n_alpha, n_beta, n_xp) < 2:
            raise ValueError("longitudinal integration counts must be at least two")
        pT2_values = np.atleast_1d(np.asarray(pT2, dtype=float))
        if np.any(pT2_values < 0.0):
            raise ValueError("pT2 must be non-negative")
        s_hadron = sqrt_s**2
        alpha_grid = np.linspace(0.05, 0.95, n_alpha)
        beta_grid = np.linspace(0.05, 0.95, n_beta)
        xP_grid = np.linspace(1.0e-3, 1.0 - xF_min, n_xp)
        da, db, dxp = (grid[1] - grid[0] for grid in (alpha_grid, beta_grid, xP_grid))
        threshold = 4.0 * self.kernel.flavor.mass**2 / ((1.0 - xF_min) * s_hadron)
        value = np.zeros_like(pT2_values)
        variance = np.zeros_like(pT2_values) if calculate_mc_error else None
        seed_offset = 0

        for x_pomeron in xP_grid:
            for alpha in alpha_grid:
                x_projectile = x_pomeron / alpha
                if not 0.0 < x_projectile < 1.0:
                    continue
                xtilde = 4.0 * self.kernel.flavor.mass**2 / (x_projectile * s_hadron)
                quarks, gluons = self.pdf.densities(x_projectile, self.kernel.flavor.mu2)
                for beta in beta_grid:
                    if alpha * beta < threshold:
                        continue
                    q_value, q_error = self.partonic_pt2(
                        alpha, beta, xtilde, "quark", pT2_values, seed_offset,
                        calculate_mc_error=calculate_mc_error,
                    )
                    g_value, g_error = self.partonic_pt2(
                        alpha, beta, xtilde, "gluon", pT2_values, seed_offset + 1,
                        calculate_mc_error=calculate_mc_error,
                    )
                    seed_offset += 2
                    coefficient = da * db * dxp / 3.0
                    value += coefficient * (quarks * q_value + 81.0 / 16.0 * gluons * g_value)
                    if calculate_mc_error:
                        variance += coefficient**2 * (
                            (quarks * q_error) ** 2 + (81.0 / 16.0 * gluons * g_error) ** 2
                        )

        conversion = survival.probability() / survival.b_sd * GEV2_TO_MB * MB_TO_UB
        value *= conversion
        result: dict[str, np.ndarray | float] = {
            "pT2": pT2_values,
            "dSigma_dpT2_ub_per_GeV2": value,
            "survival_probability": survival.probability(),
        }
        if calculate_mc_error:
            result["mc_error_ub_per_GeV2"] = np.sqrt(variance) * conversion
        return result


def paper_figure10_setups() -> dict[float, GapSurvival]:
    """Starting survival inputs for the Fig. 10 energies.

    The paper explicitly quotes K=0.14 at 1.8 TeV.  The 0.5 and 14 TeV rows
    are transparent eikonal starting points, not fitted constants; vary them
    in the uncertainty study rather than treating them as measurements.
    """
    return {
        500.0: GapSurvival(sigma_tot_mb=61.0, b_el=15.0, b_sd=11.0),
        1800.0: GapSurvival(sigma_tot_mb=80.0, b_el=17.0, b_sd=13.0),
        14000.0: GapSurvival(sigma_tot_mb=110.0, b_el=20.0, b_sd=13.0),
    }


def figure10_charm_curves(pdf: PDFProvider, pT2: np.ndarray | None = None,
                          settings: IntegrationSettings = IntegrationSettings(),
                          n_alpha: int = 4, n_beta: int = 4,
                          n_xp: int = 4) -> dict[float, dict[str, np.ndarray | float]]:
    """Compute the three charm curves shown in Fig. 10 (0.5, 1.8, 14 TeV).

    The returned ordinate is in microbarn/GeV^2; divide by 1000 for the
    mb/GeV^2 units used in the paper.  ``n_alpha``, ``n_beta``, and ``n_xp``
    control the longitudinal integrations in Eq. (72).  A real LHAPDF
    provider is required for a prediction; ``ToyPDF`` is diagnostic only.
    """
    if pT2 is None:
        pT2 = np.linspace(0.25, 20.0, 32)
    calculator = DiffractivePairCalculator(
        HeavyFlavorKernel(CHARM, GBWParameters()), pdf=pdf, settings=settings
    )
    return {
        energy: calculator.pp_pt2_spectrum(
            pT2,
            energy,
            survival,
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_xp=n_xp,
            calculate_mc_error=False,
        )
        for energy, survival in paper_figure10_setups().items()
    }


def plot_figure10(curves: dict[float, dict[str, np.ndarray | float]], output_path: str = "figure10_charm.pdf") -> None:
    """Plot the Figure-10 energy curves without Monte Carlo error bands."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)  # save files without Tcl/Tk
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency diagnostic
        raise ImportError("Plotting needs matplotlib: python -m pip install matplotlib") from exc

    fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for energy, result in sorted(curves.items(), reverse=True):
        x = np.asarray(result["pT2"])
        y = np.asarray(result["dSigma_dpT2_ub_per_GeV2"]) / 1000.0
        axis.plot(x, y, label=fr"$\sqrt{{s}}={energy / 1000:g}$ TeV")
    axis.set_yscale("log")
    axis.set_xlabel(r"$p_T^2\;[\mathrm{GeV}^2]$")
    axis.set_ylabel(r"$d\sigma_{\mathrm{diff}}/dp_T^2\;[\mathrm{mb}/\mathrm{GeV}^2]$")
    axis.set_title("Diffractive charm production - production mechanism")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
