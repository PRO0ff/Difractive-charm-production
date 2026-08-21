#!/usr/bin/env bash
# Run one of the project's Python scripts using its locked Linux environment.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 SCRIPT.py [SCRIPT ARGUMENTS...]" >&2
    echo "Example: $0 plot_charm_parameter_study.py" >&2
    exit 2
fi

uv sync --frozen
exec uv run python "$@"
