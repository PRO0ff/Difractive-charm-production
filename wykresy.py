"""Generate Figure-10-style charm curves using the CT18NLO LHAPDF set.

This is a physics-prediction script and requires the LHAPDF Python bindings
and the CT18NLO data set.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from diffractive_heavy_flavor import (
    CHARM,
    IntegrationSettings,
    LHAPDFProvider,
    figure10_charm_curves,
    plot_figure10,
)


def choose_pdf() -> LHAPDFProvider:
    """Load the real CT18NLO PDF set required for this prediction."""
    try:
        return LHAPDFProvider("CT18NLO", nf=CHARM.nf)
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "CT18NLO via LHAPDF is required. Install the LHAPDF Python "
            "bindings and the CT18NLO PDF data set, then run this script again."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figure10_charm.pdf"))
    parser.add_argument(
        "--samples",
        type=int,
        default=2**17,
        help="Monte-Carlo samples per pT integral (default: 131072)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("samples must be at least 2")

    pdf = choose_pdf()
    start_time = time.perf_counter()
    print("Calculating curves with CT18NLO...", flush=True)

    curves = figure10_charm_curves(
        pdf,
        pT2=np.linspace(0.25, 20.0, 48),
        settings=IntegrationSettings(n_samples=args.samples),
        n_alpha=5,
        n_beta=5,
        n_xp=5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_figure10(curves, str(args.output))

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Generated: {args.output.resolve()}")
    print("PDF input: CT18NLO")
    print(
        f"Finished in {elapsed_seconds:.1f} seconds "
        f"({elapsed_seconds / 60.0:.2f} minutes)."
    )


if __name__ == "__main__":
    main()
