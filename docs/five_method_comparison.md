# Five-method TSV comparison

> Historical experiment record: figures moved to `archive/five_method_figures/`.
> The old diagnostic collector used a separate reconstruction path; its TSV-M diagnostic
> must not be treated as evidence for the canonical fixed-slot TSV-M kernel.
> Current collection exports existing run traces only, and plotting reads saved records.
> Old commands and descriptions below are retained as history; use [the current README](../README.md).

This experiment compares the existing `TSVM` and `AdaptiveTSV` methods with
three new component-selection methods. The existing `zeroshot`, `average`,
`sum`, and `TSVM` merge implementations are not changed.

## Methods

All methods apply economy SVD to each two-dimensional task-vector parameter,
concatenate selected singular components in task order, compute SVD-based polar
factors for the concatenated left and right vectors, and reconstruct with the
original singular values.

* `TSVM`: selects `floor(r / T)` components from every active task.
* `AdaptiveTSV`: preserves the TSV-M layer budget and reallocates it from a
  95% normalized-energy initialization.
* `GlobalEnergyTopR`: selects the globally largest `r` task-normalized singular
  energy components per layer.
* `PerTask95FullConcat`: selects every component needed for 95% energy in each
  task without reducing the concatenated width.
* `FullComponentFullConcat`: selects every economy-SVD component from every
  active task without reducing the concatenated width.

The energy threshold is `0.95`, the numerical task-vector energy tolerance is
`1e-12`, and the reconstruction always uses the unnormalized singular values.
Non-matrix parameters retain the existing task-wise averaging behavior.

## Classification runs

Existing TSV-M and Adaptive TSV-M result JSON files may be reused. Run only the
three new methods when their result files do not yet exist:

```bash
HYDRA_FULL_ERROR=1 python main.py model=ViT-B-32 num_tasks=8 method=GlobalEnergyTopR wandb.mode=disabled save_subfolder=five_method_comparison
HYDRA_FULL_ERROR=1 python main.py model=ViT-B-32 num_tasks=8 method=PerTask95FullConcat wandb.mode=disabled save_subfolder=five_method_comparison
HYDRA_FULL_ERROR=1 python main.py model=ViT-B-32 num_tasks=8 method=FullComponentFullConcat wandb.mode=disabled save_subfolder=five_method_comparison
```

Each run evaluates alpha values from `0.1` through `3.0` in increments of
`0.1`. It selects alpha by average normalized validation accuracy and evaluates
the test datasets once with that fixed alpha. Test results are not used for
selection.

Use a new `save_subfolder` for a repeated run rather than overwriting an
existing research result.

## Common diagnostics

The plotting diagnostics require the same implementation path for all five
methods. This command recomputes merging only; it does not rerun classification:

```bash
python scripts/collect_five_method_merge_diagnostics.py
```

It writes
`results/five_method_comparison/ViT-B-32_8tasks_merge_diagnostics.json` and
refuses to overwrite an existing file unless `--overwrite` is explicitly
given. The JSON contains layer/task ranks, layer and cross-layer selection
energy, normalized concatenated width, Procrustes deformation, merge time, and
peak allocated GPU memory.

## Figures

After the three result JSON files and common diagnostics JSON exist, run:

```bash
micromamba run -n tsv-manim python scripts/make_five_method_figures.py
```

The command writes figures 1 through 11 as PNG and SVG files under
`five_method_figures/`, plus one source CSV per figure under
`five_method_figures/data/`. It does not generate a cumulative singular-energy
curve. Figure 10 is skipped only when GPU memory data is unavailable.
