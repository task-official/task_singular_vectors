# ViT-B-32 Eight-Task Reproduction

> Current entrypoints: the commands below record the historical reproduction.
> `main.py` now translates into the common runner, requires `prepare.py`, and writes
> exclusive new-format run directories beneath the requested `results/<save_subfolder>`.
> It does not overwrite the historical result filenames or reuse the old normalization JSON.
> Historical metadata and single-task logs moved to `archive/repro_metadata/`,
> `archive/reproduction_metadata/`, and `archive/run_logs/`. See [the current README](../README.md).

This document records the local checkpoint-based reproduction of the
eight-task ViT-B-32 experiments from:

> *Task Singular Vectors: Reducing Task Interference in Model Merging*

* Paper reference: arXiv `2412.00081v3`
* Upstream repository: `AntoAndGar/task_singular_vectors`
* Pinned upstream commit:
  `a6c97188f7aa0bb20753de4f2939be89dd64f1c3`

This repository is an independent reproduction and research fork. It is
not the official upstream repository.

## Scope

The current reproduction evaluates author-provided fine-tuned checkpoints.

* Model: `ViT-B-32`
* Number of tasks: `8`
* Training from scratch: not reproduced
* Device: CUDA
* W&B: disabled
* Hyperparameter selection: validation set only

The eight tasks are:

1. Cars
2. DTD
3. EuroSAT
4. GTSRB
5. MNIST
6. RESISC45
7. SVHN
8. SUN397

## Environment

The reproduction was performed in the following environment:

* Platform: WSL2
* Operating system: Ubuntu 24.04
* Environment manager: micromamba
* Environment name: `tsv`
* Python: 3.10
* GPU execution: CUDA

The repository environment should be created from the upstream
`environment.yml` when possible.

Activate the environment before running evaluations:

```bash
micromamba activate tsv
cd ~/work/TSV-Merge
```

Verify CUDA availability:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

## Checkpoints

Author-provided checkpoints are expected under:

```text
models/checkpoints/
```

The exact filenames and checksums should be verified locally before
evaluation.

Checkpoint files are not committed to Git.

## Dataset locations

The evaluation configuration uses:

```yaml
data_location: datasets
model_location: models/checkpoints
```

The relevant local dataset locations are:

```text
datasets/stanford_cars/
datasets/dtd/
datasets/EuroSAT_splits/
datasets/resisc45/
datasets/sun397/
datasets/svhn/
```

Some torchvision-managed datasets, including MNIST and GTSRB, may use
subdirectories under `datasets/`.

Dataset files are not committed to Git.

## Dataset-specific notes

### Cars

The expected structure includes:

```text
datasets/stanford_cars/
├── devkit/
│   ├── cars_meta.mat
│   └── cars_train_annos.mat
├── cars_train/
├── cars_test/
└── cars_test_annos_withlabels.mat
```

### EuroSAT

The local reproduction uses a generated stratified split:

* train/validation/test ratio: `80/10/10`
* seed: `42`

Identity with the authors' exact split has not been verified.

### RESISC45

The dataset contains 45 classes with 700 images per class.

The local reproduction uses a generated stratified split:

* train/validation/test ratio: `80/10/10`
* seed: `42`

Identity with the authors' exact split has not been verified.

### SUN397

The local data was reconstructed from the Hugging Face dataset:

```text
tanganke/sun397
```

The exported dataset was checked for:

* 397 classes
* 19,850 training images
* 19,850 validation images
* 50 images per class in each split

Complete byte-level identity with the original SUN397 archive has not
been verified.

## Single-task normalization evaluation

The local evaluator limits evaluation to the eight tasks used in this
reproduction:

```bash
python eval_single_task_8.py \
  --model=ViT-B-32 \
  --finetuning-mode=standard \
  2>&1 | tee results/eval_single_task_8_ViT-B-32.log
```

The normalization results are written to:

```text
results/single_task/ViT-B-32/nonlinear_ft_accuracies.json
```

This file is produced by evaluation code, not by the documentation
generation process.

## Zero-shot evaluation

```bash
HYDRA_FULL_ERROR=1 python main.py \
  model=ViT-B-32 \
  num_tasks=8 \
  method=zeroshot \
  wandb.mode=disabled \
  save_subfolder=repro_vitb32_8
```

## Weight Averaging

```bash
HYDRA_FULL_ERROR=1 python main.py \
  model=ViT-B-32 \
  num_tasks=8 \
  method=average \
  wandb.mode=disabled \
  save_subfolder=repro_vitb32_8
```

## Task Arithmetic

```bash
HYDRA_FULL_ERROR=1 python main.py \
  model=ViT-B-32 \
  num_tasks=8 \
  method=sum \
  wandb.mode=disabled \
  save_subfolder=repro_vitb32_8
```

The scaling coefficient is selected using validation normalized
accuracy. Test results must not be used to select the coefficient.

## TSV-Merge

```bash
HYDRA_FULL_ERROR=1 python main.py \
  model=ViT-B-32 \
  num_tasks=8 \
  method=TSVM \
  wandb.mode=disabled \
  save_subfolder=repro_vitb32_8
```

The existing `TSVM` method is treated as the protected reproduction
baseline. Research extensions should use a different method name or
configuration instead of silently changing the baseline behavior.

## Results

The currently recorded results are:

| Method           | Paper absolute (%) | Paper normalized (%) | Local absolute (%) | Local normalized (%) | Selected coefficient |
| ---------------- | -----------------: | -------------------: | -----------------: | -------------------: | -------------------: |
| Zero-shot        |              48.26 |                53.59 |              48.42 |                53.50 |                  0.0 |
| Weight Averaging |              66.34 |                72.13 |              66.68 |                72.16 |                0.125 |
| Task Arithmetic  |              70.79 |                76.55 |              71.68 |                77.01 |               0.2333 |
| TSV-Merge        |              85.86 |                92.31 |              86.48 |                92.56 |                  1.0 |

For TSV-Merge, the coefficient was selected by maximizing validation
normalized accuracy over the evaluated values from `0.1` to `3.0`.

The selected coefficient was `1.0`, for which the validation metrics
were:

* validation absolute accuracy: `87.06%`
* validation normalized accuracy: `90.86%`

The resulting test metrics were:

* test absolute accuracy: `86.48%`
* test normalized accuracy: `92.56%`

The source result files currently include:

```text
results/repro_vitb32_8/ViT-B-32_8tasks_zeroshot_k=100_nonlinear_additions_.json
results/repro_vitb32_8/ViT-B-32_8tasks_average_k=100_nonlinear_additions_.json
results/repro_vitb32_8/ViT-B-32_8tasks_sum_k=100_nonlinear_additions_.json
results/repro_vitb32_8/ViT-B-32_8tasks_TSVM_k=100_nonlinear_additions_svd.json
```

The TSV-Merge values above were read from:

```text
test.avg_top1
test.avg_normalized_top1
```

in:

```text
results/repro_vitb32_8/ViT-B-32_8tasks_TSVM_k=100_nonlinear_additions_svd.json
```

The selected coefficient was identified by matching `val_best` to the
corresponding entry under `val`, which is `val["1.0"]`.

Before publishing final results, all reported values should be verified
directly against the source JSON files.

The local values are not treated as evidence of a strict improvement
over the paper because dataset sources and split definitions may differ
from those used by the authors.

## Local changes relevant to reproduction

### Eight-task evaluator

`eval_single_task_8.py` evaluates only the eight datasets used in the
target experiment. The upstream evaluator is retained separately.

### Device and dtype alignment

`src/models/task_vectors.py` contains a local compatibility change that
aligns task-vector tensors with the pretrained parameter device and
dtype before addition.

This change should be reviewed separately from documentation changes.

### W&B configuration

Evaluation commands use:

```text
wandb.mode=disabled
```

This avoids requiring online W&B logging during local reproduction.

## Known deviations and unresolved differences

The following differences or uncertainties remain:

* Fine-tuning from scratch has not been reproduced.
* Exact identity of all local dataset splits with the authors' splits is
  not confirmed.
* EuroSAT uses a locally generated stratified split.
* RESISC45 uses a locally generated stratified split.
* SUN397 was reconstructed from `tanganke/sun397`.
* Complete byte identity of the SUN397 files with the original archive
  is not confirmed.
* Exact identity of every checkpoint with the artifacts used for the
  paper tables has not been independently verified.
* The effect of the local device/dtype compatibility change should be
  reported when comparing against the unmodified upstream
  implementation.

## Repository hygiene

Do not commit:

* dataset contents
* model checkpoints
* local paper PDFs
* credentials or API keys
* raw W&B data
* large raw logs
* machine-specific absolute paths

Small scripts and concise result summaries may be committed after
review.
