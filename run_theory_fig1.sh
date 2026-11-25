#!/usr/bin/env bash
# Run both Figure 1 configurations using the active Python environment.
# Override the interpreter with PYTHON=/path/to/python; extra arguments are
# forwarded to both runs (for example, --validate-only or --fork-steps 20).
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"

for config in fig1_config.example_max_corr.json fig1_config.example.json; do
    printf '\nRunning Figure 1: %s\n' "$config"
    "$python_bin" "$repo_dir/theory_check/run_fig1.py" \
        --config "$repo_dir/theory_check/$config" "$@"
done
