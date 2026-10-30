# Instructions for Coding Agents

## Project identity

This repository is a reproduction and research-extension fork of:

`https://github.com/AntoAndGar/task_singular_vectors`

The reproduction baseline is pinned to upstream commit:

`a6c97188f7aa0bb20753de4f2939be89dd64f1c3`

This is not the official upstream repository.

## Read before modifying code

Before changing implementation or evaluation behavior, read:

1. `README.md`
2. `docs/reproduction.md`
3. the relevant source files and configuration files

Do not assume that local datasets, checkpoints, result files, or logs are
available in Git. These are intentionally excluded or may exist only in
the local workspace.

## Protected reproduction baseline

The documented reproduction baseline uses:

* model: `ViT-B-32`
* number of tasks: `8`
* checkpoints: author-provided fine-tuned checkpoints
* methods:

  * `zeroshot`
  * `average`
  * `sum`
  * `TSVM`

Do not silently change the behavior of these existing method names.

When implementing experimental behavior, prefer a new method name,
configuration option, or clearly separated code path.

## Scientific rules

* Do not use test results for hyperparameter selection.
* Select coefficients and other tunable parameters using validation data.
* Keep paper values separate from locally reproduced values.
* Report known dataset, split, checkpoint, and implementation differences.
* Do not describe a local result as an exact reproduction unless the
  relevant artifacts and conditions have been verified.
* Preserve source result JSON files and raw experimental evidence.

## Repository safety

Do not commit:

* dataset contents
* model checkpoints
* credentials, tokens, or API keys
* local paper PDFs
* raw W&B data
* large raw logs
* machine-specific absolute paths
* temporary downloads, caches, or editor state

Do not delete, overwrite, restore, or reformat research results merely as
part of a documentation or cleanup task.

In particular, treat these as research artifacts unless the user
explicitly requests otherwise:

* `results/`
* `logs/`
* `run_logs/`
* `repro_metadata/`
* `reproduction_metadata/`
* research and dataset-processing scripts under `scripts/`

## Local implementation changes

The current workspace may contain local changes used for reproduction,
including:

* `eval_single_task_8.py`
* `src/models/task_vectors.py`

Review such code changes separately from documentation changes.

Do not revert them solely because they differ from the pinned upstream
commit.

## Change discipline

* Inspect the current Git status before editing.
* Make the smallest change required for the requested task.
* Do not create additional documentation frameworks, manifests, ADR
  systems, run registries, or bootstrap generators unless explicitly
  requested.
* Keep documentation consistent with files that actually exist.
* Verify commands and paths against the current repository before adding
  them to documentation.
* Separate documentation changes, implementation changes, and result
  updates when preparing commits.
