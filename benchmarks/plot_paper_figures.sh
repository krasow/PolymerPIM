#!/usr/bin/env bash

set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python3}"

"${python}" "${dir}/plot_weak_scaling.py" # figure 3
"${python}" "${dir}/plot_runtime_decomposition.py" # figure 4
"${python}" "${dir}/plot_fusion_traces.py" # figure 5
"${python}" "${dir}/plot_param_gain.py" # figure 6
PLOT_SUITE=modes "${python}" "${dir}/plot_weak_scaling.py" # figure 7
"${python}" "${dir}/dynamic-benchmarks/plot_query_sweep.py" # figure 8
"${python}" "${dir}/dynamic-benchmarks/plot_adaptive_image.py" # figure 9
