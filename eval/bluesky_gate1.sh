#!/usr/bin/env bash
# Run upstream verifiers without writing into either checkout.
set -euo pipefail

plan=${1:?pass a or b}
checkout=$(realpath -- "${2:?pass the BlueSky checkout}")
output=$(realpath -m -- "${3:?pass a new output directory}")
python_bin=${PYTHON_BIN:-python}
project=$(cd -- "$(dirname -- "$0")/.." && pwd)
case "$plan" in
  a|b) ;;
  *) printf 'Unknown plan %s\n' "$plan" >&2; exit 2 ;;
esac

mkdir "$output"
mkdir "$output/tmp" "$output/cache"
git -C "$checkout" rev-parse HEAD > "$output/commit.txt"
git -C "$checkout" status --porcelain > "$output/checkout-status-before.txt"
export PYTHONPATH="$checkout:$project" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$output/tmp" XDG_CACHE_HOME="$output/cache" MPLCONFIGDIR="$output/cache/mpl"
export HF_HOME="$output/cache/hf" TORCH_HOME="$output/cache/torch" PIP_CACHE_DIR="$output/cache/pip"
cd "$output"
test_rc=0

measure() {
  local label=$1 rc=0
  shift
  printf 'Running %s\n' "$label"
  "$@" > "$label.log" 2>&1 || rc=$?
  printf '%s\t%s\n' "$label" "$rc" | tee -a command-exit-status.tsv
  cat "$label.log"
  if (( rc != 0 )); then test_rc=1; fi
}

"$python_bin" -B -c 'import sys; print(sys.version)' > python.txt
runner=("$python_bin" -B -m eval.run_bluesky "$output/bluesky")
if [[ "$plan" == a ]]; then
  measure benchmark-a "${runner[@]}" bluesky.rl.benchmark_plan_a --steps 20 --ac 12 --sim-dt 0.05
  measure pytest-a "${runner[@]}" pytest -q -p no:cacheprovider "$checkout/tests/test_plan_a_batched_env.py"
  mkdir "$output/parity-a"
  for mode in native fast packed; do
    measure "trace-$mode" "$python_bin" -B -m eval.verify_plan_a "$mode" "$output/parity-a"
  done
  measure parity-a "$python_bin" -B -m eval.verify_plan_a compare "$output/parity-a"
else
  ppu-smi > ppu-before.txt
  measure parity "${runner[@]}" bluesky.rl_gpu.verify_parity
  measure benchmark-b "${runner[@]}" bluesky.rl_gpu.benchmark_plan_b --device cuda --steps 50 --ac 16 --sim-dt 0.25
  measure pytest-b "${runner[@]}" pytest -q -p no:cacheprovider "$checkout/tests/test_plan_b_torch_kernel.py"
  ppu-smi > ppu-after.txt
fi
git -C "$checkout" status --porcelain > checkout-status-after.txt
exit "$test_rc"
