"""Parameter scans for diffractive charm-pair production.

This script makes three reproducible diagnostic figures from the production
mechanism in ``diffractive_heavy_flavor.py``:

* ``charm_model_parameter_summary.png`` scans GBW, alpha_s, m_c and the
  gap-survival inputs;
* ``charm_partonic_alpha_beta.png`` shows the deterministic forward partonic
  cross sections across the (alpha, beta) phase space;
* ``charm_pt2_by_energy.png`` is a Figure-10-style pp pT^2 scan, including
  Monte-Carlo error bands.

The first two figures use the verified partonic kernel and are independent of
projectile PDFs.  The pp pT^2 figure needs PDFs.  When LHAPDF/CT18NLO is not
available the script deliberately falls back to ToyPDF and labels the result
as a diagnostic, not an absolute physics prediction.

Run from the project root:

    .venv/bin/python plot_charm_parameter_study.py

For a higher-statistics pT^2 plot, increase --pt-samples and the three
longitudinal integration counts together, then check that both values and
error bands are stable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from diffractive_heavy_flavor import (
    CHARM,
    FM_TO_GEVINV,
    GEV2_TO_MB,
    GBWParameters,
    GapSurvival,
    HeavyFlavor,
    HeavyFlavorKernel,
    IntegrationSettings,
    LHAPDFProvider,
    ToyPDF,
    DiffractivePairCalculator,
    paper_figure10_setups,
)


SQRT_S_GEV = 13_600.0
X_PROJECTILE = 0.05
ALPHA_REFERENCE = 0.5
BETA_REFERENCE = 0.5


def xtilde_for_charm(mass: float, sqrt_s: float = SQRT_S_GEV,
                      x_projectile: float = X_PROJECTILE) -> float:
    """Dipole Bjorken-x used in the paper's production mechanism."""
    return 4.0 * mass**2 / (x_projectile * sqrt_s**2)


def make_calculator(mass: float = CHARM.mass,
                    settings: IntegrationSettings | None = None) -> DiffractivePairCalculator:
    """Build a deterministic forward calculator with a placeholder PDF."""
    flavor = HeavyFlavor("charm", mass=mass, nf=3)
    kernel = HeavyFlavorKernel(flavor, GBWParameters())
    return DiffractivePairCalculator(kernel, ToyPDF(), settings or IntegrationSettings())


def choose_pdf() -> tuple[object, str]:
    """Use CT18NLO when installed; otherwise make the diagnostic status explicit."""
    try:
        return LHAPDFProvider("CT18NLO", nf=CHARM.nf), "CT18NLO"
    except (ImportError, RuntimeError, OSError):
        return ToyPDF(), "ToyPDF (diagnostic normalization)"


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> Path:
    """Save a consistently sized, high-resolution raster figure."""
    output_path = output_dir / f"{stem}.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def plot_model_parameter_summary(output_dir: Path, settings: IntegrationSettings) -> Path:
    """Plot the model ingredients and their direct effect on the partonic rate."""
    gbw = GBWParameters()
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), layout="constrained")

    # GBW saturation cross section for the x range that appears in the folding.
    radius_fm = np.geomspace(0.003, 1.2, 300)
    radius = radius_fm * FM_TO_GEVINV
    for xtilde, colour in zip((1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4),
                              ("#332288", "#44AA99", "#DDCC77", "#CC6677"), strict=True):
        sigma_mb = gbw.sigma(radius, xtilde) * GEV2_TO_MB
        r0_fm = gbw.r0(xtilde) / FM_TO_GEVINV
        axes[0, 0].plot(radius_fm, sigma_mb, color=colour,
                         label=rf"$\tilde{{x}}={xtilde:.0e}$, $R_0={r0_fm:.2f}$ fm")
    axes[0, 0].set(xscale="log", xlabel=r"dipole size $r$ [fm]",
                   ylabel=r"$\sigma_{q\bar q}(r)$ [mb]",
                   title="GBW dipole cross section")
    axes[0, 0].legend(fontsize=8, frameon=False)

    # Infrared freezing of alpha_s.
    scale = np.geomspace(0.15, 30.0, 400)
    for frozen, colour in zip((0.35, 0.40, 0.50), ("#4477AA", "#228833", "#CC6677"), strict=True):
        kernel = HeavyFlavorKernel(CHARM, gbw, alpha_s_frozen=frozen)
        axes[0, 1].plot(scale, kernel.alpha_s(scale), color=colour,
                         label=rf"$\alpha_s^{{\rm max}}={frozen:.2f}$")
    axes[0, 1].set(xscale="log", xlabel=r"scale $Q$ [GeV]", ylabel=r"$\alpha_s(Q)$",
                   title="Running coupling with infrared freezing", ylim=(0.0, 0.55))
    axes[0, 1].legend(frameon=False)

    # The partonic mass scan is fully deterministic and uses the same point in phase space.
    masses = np.linspace(1.2, 1.8, 13)
    partonic_q = []
    partonic_g = []
    for mass in masses:
        calculator = make_calculator(mass, settings)
        xtilde = xtilde_for_charm(mass)
        partonic_q.append(calculator.partonic_forward(
            ALPHA_REFERENCE, BETA_REFERENCE, xtilde, "quark"
        ))
        partonic_g.append(calculator.partonic_forward(
            ALPHA_REFERENCE, BETA_REFERENCE, xtilde, "gluon"
        ))
    axes[1, 0].plot(masses, partonic_q, marker="o", ms=3.5, label="quark projectile")
    axes[1, 0].plot(masses, partonic_g, marker="s", ms=3.5, label="gluon projectile")
    axes[1, 0].set(yscale="log", xlabel=r"charm mass $m_c$ [GeV]",
                   ylabel=r"$d\sigma/(dt'\,d\alpha\,d\beta)$ [GeV$^{-4}$]",
                   title=rf"Partonic mass scan at $\alpha=\beta=0.5$, $x_p={X_PROJECTILE:g}$")
    axes[1, 0].legend(frameon=False)

    # After t' integration the normalization is proportional to K / B_sd.
    b_sd = np.linspace(8.0, 20.0, 160)
    baseline = GapSurvival(sigma_tot_mb=115.0, b_el=21.0, b_sd=13.0)
    baseline_scale = baseline.probability() / baseline.b_sd
    for sigma_tot_mb, colour in zip((110.0, 115.0, 120.0),
                                    ("#4477AA", "#228833", "#CC6677"), strict=True):
        scale_factor = np.array([
            GapSurvival(sigma_tot_mb=sigma_tot_mb, b_el=21.0, b_sd=value).probability() / value
            for value in b_sd
        ]) / baseline_scale
        axes[1, 1].plot(b_sd, scale_factor, color=colour,
                         label=rf"$\sigma_{{\rm tot}}={sigma_tot_mb:.0f}$ mb")
    axes[1, 1].axvline(13.0, color="0.45", ls="--", lw=1)
    axes[1, 1].axhline(1.0, color="0.45", ls=":", lw=1)
    axes[1, 1].set(xlabel=r"diffractive slope $B_{\rm sd}$ [GeV$^{-2}$]",
                   ylabel=r"relative pp normalization $(K/B_{\rm sd})/(K/B_{\rm sd})_0$",
                   title="Gap-survival normalization at 13.6 TeV")
    axes[1, 1].legend(frameon=False)

    figure.suptitle("Diffractive charm: parameter dependence of the production mechanism", fontsize=14)
    return save_figure(figure, output_dir, "charm_model_parameter_summary")


def plot_alpha_beta_maps(output_dir: Path, settings: IntegrationSettings, points: int) -> Path:
    """Map the quark and gluon forward kernels over alpha and beta."""
    calculator = make_calculator(settings=settings)
    alpha_values = np.linspace(0.04, 0.96, points)
    beta_values = np.linspace(0.04, 0.96, points)
    alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values, indexing="xy")
    cross_sections: dict[str, np.ndarray] = {}
    xtilde = xtilde_for_charm(CHARM.mass)
    for channel in ("quark", "gluon"):
        values = np.empty_like(alpha_grid)
        for row, beta in enumerate(beta_values):
            for column, alpha in enumerate(alpha_values):
                values[row, column] = calculator.partonic_forward(alpha, beta, xtilde, channel)
        cross_sections[channel] = values

    lower = min(array.min() for array in cross_sections.values())
    upper = max(array.max() for array in cross_sections.values())
    norm = colors.LogNorm(vmin=lower, vmax=upper)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), layout="constrained", sharey=True)
    image = None
    for axis, channel in zip(axes, ("quark", "gluon"), strict=True):
        image = axis.pcolormesh(alpha_grid, beta_grid, cross_sections[channel], shading="auto",
                                cmap="viridis", norm=norm)
        axis.plot([0.04, 0.96], [0.5, 0.5], color="white", lw=0.75, alpha=0.7)
        axis.axvline(0.5, color="white", lw=0.75, alpha=0.7)
        axis.set(xlabel=r"pair momentum fraction $\alpha$", title=f"{channel.capitalize()} projectile")
    axes[0].set_ylabel(r"charm momentum fraction $\beta$")
    colourbar = figure.colorbar(image, ax=axes, shrink=0.91, pad=0.025)
    colourbar.set_label(r"$d\sigma/(dt'\,d\alpha\,d\beta)$ [GeV$^{-4}$]")
    figure.suptitle(
        rf"Forward diffractive charm kernel at $\sqrt{{s}}={SQRT_S_GEV / 1000:g}$ TeV and $x_p={X_PROJECTILE:g}$"
    )
    return save_figure(figure, output_dir, "charm_partonic_alpha_beta")


def plot_pt2_by_energy(output_dir: Path, settings: IntegrationSettings,
                       n_longitudinal: int, n_pt2: int) -> tuple[Path, str]:
    """Make a Figure-10-style pT^2 plot from the off-diagonal kernel."""
    pdf, pdf_label = choose_pdf()
    kernel = HeavyFlavorKernel(CHARM, GBWParameters())
    calculator = DiffractivePairCalculator(kernel, pdf, settings=settings)
    p_t2 = np.linspace(0.25, 20.0, n_pt2)
    figure, axis = plt.subplots(figsize=(7.7, 5.2), layout="constrained")
    for energy, survival in sorted(paper_figure10_setups().items(), reverse=True):
        result = calculator.pp_pt2_spectrum(
            p_t2, energy, survival,
            n_alpha=n_longitudinal, n_beta=n_longitudinal, n_xp=n_longitudinal,
        )
        ordinate = np.asarray(result["dSigma_dpT2_ub_per_GeV2"]) / 1_000.0
        error = np.asarray(result["mc_error_ub_per_GeV2"]) / 1_000.0
        positive = ordinate > 0.0
        label = rf"$\sqrt{{s}}={energy / 1_000:g}$ TeV, $K={result['survival_probability']:.2f}$"
        axis.plot(p_t2[positive], ordinate[positive], lw=2.0, label=label)
        axis.fill_between(
            p_t2[positive],
            np.maximum(ordinate[positive] - error[positive], 1.0e-30),
            ordinate[positive] + error[positive],
            alpha=0.18,
        )
    axis.set(
        yscale="log",
        xlabel=r"$p_T^2$ [GeV$^2$]",
        ylabel=r"$d\sigma_{\rm diff}/dp_T^2$ [mb/GeV$^2$]",
        title="Diffractive charm pT spectrum - production mechanism",
    )
    axis.legend(fontsize=8.5, frameon=False)
    if pdf_label.startswith("ToyPDF"):
        axis.text(
            0.02, 0.02, "ToyPDF: diagnostic shape and normalization only",
            transform=axis.transAxes, fontsize=8.5, color="#8b1a1a",
            bbox={"facecolor": "white", "edgecolor": "#8b1a1a", "alpha": 0.9, "pad": 3},
        )
    return save_figure(figure, output_dir, "charm_pt2_by_energy"), pdf_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/charm_parameter_study"))
    parser.add_argument("--map-points", type=int, default=25,
                        help="number of alpha/beta points per axis (default: 25)")
    parser.add_argument("--pt-samples", type=int, default=2**15,
                        help="Monte-Carlo samples per pT integral (default: 32768)")
    parser.add_argument("--longitudinal-points", type=int, default=4,
                        help="points for each pp longitudinal integral (default: 4)")
    parser.add_argument("--pt2-points", type=int, default=28,
                        help="number of pT^2 points in the pp spectrum (default: 28)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.map_points < 3 or args.longitudinal_points < 2 or args.pt2_points < 3:
        raise ValueError("map-points >= 3, longitudinal-points >= 2, and pt2-points >= 3 are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deterministic_settings = IntegrationSettings(n_rho=48, n_s=48, n_phi=36, n_samples=2**10)
    partonic_summary = plot_model_parameter_summary(args.output_dir, deterministic_settings)
    alpha_beta_map = plot_alpha_beta_maps(args.output_dir, deterministic_settings, args.map_points)
    pt_settings = IntegrationSettings(n_samples=args.pt_samples)
    pt2_plot, pdf_label = plot_pt2_by_energy(
        args.output_dir, pt_settings, args.longitudinal_points, args.pt2_points
    )

    print("Generated:")
    for path in (partonic_summary, alpha_beta_map, pt2_plot):
        print(f"  {path.resolve()}")
    print(f"PDF input for pp pT^2 plot: {pdf_label}")


if __name__ == "__main__":
    main()
