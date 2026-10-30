"""Explicit local preparation and preflight; no dataset/network side effects."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from src.core.checkpoints import file_sha256, make_inputs, open_prepared, prepare_checkpoint
from src.core.types import ExperimentSpec, PreparedExperiment


def prepare_inputs(spec: ExperimentSpec) -> dict:
    """Prepare one checkpoint at a time; preserve author files and existing results."""
    paths = [spec.base_path, *spec.checkpoint_paths.values()]
    missing = [path for path in paths if not Path(path).expanduser().is_file()]
    missing += [task.head for task in spec.tasks if not Path(task.head).expanduser().is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    prepared = {}
    for path in dict.fromkeys(paths):
        prepared[path] = str(prepare_checkpoint(path, spec.runtime.cache_dir))
    return prepared


def resolve_inputs(spec: ExperimentSpec) -> PreparedExperiment:
    if spec.runtime.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set runtime.device=cpu explicitly for CPU execution")
    head_ids = {}
    for task in spec.tasks:
        head = Path(task.head).expanduser()
        if not head.is_file():
            raise FileNotFoundError(f"Classification head missing: {head}; prepare inputs before running")
        head_ids[task.id] = file_sha256(head)
        from src.datasets.adapter import DATASET_ADAPTERS
        for split in (task.validation, task.test):
            if split.get("adapter", "legacy") not in DATASET_ADAPTERS:
                raise ValueError(f"Unknown dataset adapter for {task.id}: {split.get('adapter')}")
    base = open_prepared(spec.base_path, spec.runtime.cache_dir)
    finetuned = {
        task.id: open_prepared(spec.checkpoint_paths[task.id], spec.runtime.cache_dir)
        for task in spec.tasks
    }
    merge_inputs = make_inputs(base, finetuned, tuple(task.id for task in spec.tasks))
    for name in merge_inputs.parameter_order:
        value = base.tensor(name)
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"Reference FP32 execution requires FP32 checkpoints: {name} is {value.dtype}; do not silently cast author artifacts")
    identity = {
        "merge_inputs": merge_inputs.identity,
        "base_sha256": base.identity,
        "finetuned_sha256": {task: source.identity for task, source in finetuned.items()},
        "heads_sha256": head_ids,
        "task_order": list(merge_inputs.task_order),
        "tasks": [asdict(task) for task in spec.tasks],
        "source_status": spec.source_status,
        # A declaration or a matching filename is not evidence of paper identity.
        "strict_reproduction_verified": False,
    }
    return PreparedExperiment(spec, merge_inputs, identity)
