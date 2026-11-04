#!/usr/bin/env bash
# The ten-method ViT-B/32 comparison documented in README.md.
# Requires prepared local inputs. Each invocation starts ten new runs;
# resume an interrupted individual run through run.py instead.
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repository_root"

experiment_id="fft8_b32_common_tuned_v1"

# Separate processes keep each method's peak RAM measurement independent.
for method_id in zeroshot weight_average task_arithmetic ties consensus_ta tsvm iso_cts dc_merge gse_tsv sera; do
  case "$method_id" in
    zeroshot|weight_average) protocol_id=fft8_tuned_single ;;
    sera) protocol_id=fft8_tuned_sera ;;
    *) protocol_id=fft8_tuned_alpha ;;
  esac
  micromamba run -n tsv python run.py \
    model=vit_b32 benchmark=fft8_all_images method="$method_id" protocol="$protocol_id" \
    runtime.device=cuda:0 runtime.compute_dtype=float32 runtime.batch_size=32 \
    runtime.num_workers=4 runtime.seed=0 runtime.trace=summary runtime.save_weights=true \
    "runtime.output_dir=results/experiments/${experiment_id}"
done
