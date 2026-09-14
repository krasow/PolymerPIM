#!/usr/bin/env bash

set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results="${dir}/results"

mkdir -p "${results}/main"

"${dir}/run.sh" --resume # figures 3, 4, 5
"${dir}/run.sh" --resume --runner --dpus 256,2048 --variant polymerpim \
    --csv "${results}/main/tuned.csv" --state "${results}/main/tuned.state.toml" # figure 6
"${dir}/run.sh" --resume --default-params --runner --dpus 256,2048 --variant polymerpim \
    --csv "${results}/main/default.csv" --state "${results}/main/default.state.toml" # figure 6
"${dir}/run.sh" --resume --default-params --config "${dir}/main-benchmarks/polymerpim-modes.toml" # figure 7
"${dir}/dynamic-benchmarks/query_sweep.sh" # figure 8
"${dir}/dynamic-benchmarks/adaptive_image.sh" # figure 9
