"""Figure-12-style diffractive top-pair pT^2 spectrum.

This uses the exact off-diagonal pT kernel in ``diffractive_heavy_flavor``
for pp -> t tbar X p at sqrt(s) = 1.8 and 14 TeV.  The momentum range follows
Fig. 12 of Kopeliovich et al. (arXiv:hep-ph/0702106).

The script selects CT18NLO through LHAPDF when it is installed.  Otherwise it
uses the project's ToyPDF only to exercise the numerical pipeline; the plot
then states clearly that its normalization is diagnostic rather than a
prediction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from diffractive_heavy_flavor import (
    GEV2_TO_MB,
    GBWParameters,
    HeavyFlavorKernel,
    IntegrationSettings,
    LHAPDFProvider,
    TOP,
    ToyPDF,
    DiffractivePairCalculator,
    paper_figure10_setups,
)


OUTPUT_DIR = Path("output/top_parameter_study")


def choose_pdf() -> tuple[object, str]:
    """Prefer a real PDF but preserve an explicit diagnostic fallback."""
    try:
        return LHAPDFProvider("CT18NLO", nf=TOP.nf), "CT18NLO"
    except (ImportError, RuntimeError, OSError):
        return ToyPDF(), "ToyPDF (diagnostic normalization)"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf, pdf_label = choose_pdf()
    calculator = DiffractivePairCalculator(
        HeavyFlavorKernel(TOP, GBWParameters()),
        pdf,
        # The Monte-Carlo error band must be considered alongside the curve.
        settings=IntegrationSettings(n_samples=2**16),
    )
    # The top panel in Fig. 12 spans roughly 2e4--1e5 GeV^2.  Include a small
    # lower extension so that the threshold-region shape is visible.
    p_t2 = np.geomspace(1.0e4, 1.2e5, 28)
    figure, axis = plt.subplots(figsize=(7.7, 5.2), layout="constrained")

    for energy in (14_000.0, 1_800.0):
        survival = paper_figure10_setups()[energy]
        result = calculator.pp_pt2_spectrum(
            p_t2,
            sqrt_s=energy,
            survival=survival,
            xF_min=0.85,
            n_alpha=4,
            n_beta=4,
            n_xp=4,
        )
        spectrum = np.asarray(result["dSigma_dpT2_ub_per_GeV2"]) / 1_000.0
        error = np.asarray(result["mc_error_ub_per_GeV2"]) / 1_000.0
        valid = spectrum > 0.0
        if not np.any(valid):
            raise RuntimeError(f"No positive top spectrum values at sqrt(s)={energy:g} GeV")
        axis.plot(
            p_t2[valid], spectrum[valid], lw=2.0,
            label=rf"$\sqrt{{s}}={energy / 1_000:g}$ TeV, $K={result['survival_probability']:.2f}$",
        )
        axis.fill_between(
            p_t2[valid],
            np.maximum(spectrum[valid] - error[valid], 1.0e-30),
            spectrum[valid] + error[valid],
            alpha=0.18,
        )

    axis.set(
        xscale="log",
        yscale="log",
        xlabel=r"$p_T^2$ [GeV$^2$]",
        ylabel=r"$d\sigma_{\rm diff}/dp_T^2$ [mb/GeV$^2$]",
        title=r"Diffractive $t\bar{t}$ $p_T^2$ spectrum - production mechanism",
    )
    axis.legend(frameon=False, fontsize=9)
    if pdf_label.startswith("ToyPDF"):
        axis.text(
            0.02, 0.02, "ToyPDF: diagnostic shape and normalization only",
            transform=axis.transAxes, fontsize=8.5, color="#8b1a1a",
            bbox={"facecolor": "white", "edgecolor": "#8b1a1a", "alpha": 0.9, "pad": 3},
        )
    output_path = OUTPUT_DIR / "top_pt2_by_energy.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(output_path.resolve())
    print(f"PDF input: {pdf_label}")


if __name__ == "__main__":
    main()
