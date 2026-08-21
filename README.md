# Diffractive charm — Linux

This project runs headlessly on Linux: its plotting scripts select the `Agg`
backend and write figures to disk, so no desktop session or display server is
needed. The lockfile includes Linux wheels for the pinned scientific Python
dependencies.

## Requirements

- Linux with Python 3.13 or later (or `uv`, which can manage Python for you)
- `uv` from <https://docs.astral.sh/uv/>
- A recent glibc-based Linux distribution for the prebuilt scientific wheels

LHAPDF and the `CT18NLO` data set are optional. Without them, the plotting
scripts use `ToyPDF` and label the resulting normalization as diagnostic.

## Run a script

From this directory:

```bash
chmod +x run_linux.sh
./run_linux.sh plot_charm_parameter_study.py
./run_linux.sh plot_beauty_pt2_by_energy.py
./run_linux.sh plot_top_pt2_by_energy.py
./run_linux.sh wykresy.py
```

`run_linux.sh` creates/synchronizes `.venv` from the exact `uv.lock` file and
then runs the script passed to it. You can also use `uv` directly:

```bash
uv sync --frozen
uv run python plot_charm_parameter_study.py
```

The charm, beauty, and top parameter-study figures are written below
`output/`. `wykresy.py` writes `figure10_charm.pdf` by default; use
`--output path/to/figure.pdf` to choose another location.

## Optional physics PDF input

For physical rather than diagnostic normalization, install the Linux LHAPDF
bindings and make the `CT18NLO` PDF data set available to LHAPDF. The precise
package name varies by Linux distribution; once installed, the scripts select
it automatically.
