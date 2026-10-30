"""An evaluator is bound to a split before a selection recipe receives it."""
from __future__ import annotations

import math
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.core.types import EvaluationResult, PreparedExperiment, RuntimeSpec, TaskMetrics
from src.datasets.adapter import EvaluationTask, file_digest, inspect_task
from src.core.numerics import seeded_rng, numerical_environment

EVALUATOR_VERSION = "macro-top1-seeded-v3-runtime-identity"


@contextmanager
def evaluation_rng(task, split, runtime):
    encoded = f"{runtime.seed}:{split}:{task.id}".encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    previous = task.loader.generator
    try:
        task.loader.generator = torch.Generator().manual_seed(seed)
        with seeded_rng(seed, runtime.device):
            yield
    finally:
        task.loader.generator = previous


class Evaluator:
    def __init__(self, adapter, tasks: Sequence[EvaluationTask], runtime: RuntimeSpec,
                 baselines: Mapping[str, float] | None = None):
        if not tasks:
            raise ValueError("An evaluator needs at least one task")
        splits = {task.split for task in tasks}
        if len(splits) != 1 or next(iter(splits)) not in ("validation", "test"):
            raise ValueError("Bind each evaluator to exactly one explicit split")
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("Duplicate task identifiers")
        self._split = next(iter(splits))
        self.adapter = adapter
        self.runtime = runtime
        self.tasks = {task.id: task for task in tasks}
        self.task_ids = tuple(self.tasks)
        self.task_sizes = {task.id: task.size for task in tasks}
        self._baselines = None if baselines is None else dict(baselines)
        if baselines is not None:
            for task_id in self.task_ids:
                if task_id not in baselines or not math.isfinite(float(baselines[task_id])) or not 0 < baselines[task_id] <= 1:
                    raise ValueError(f"Missing or nonpositive normalization baseline for {task_id}")
        self.identity = {
            "version": EVALUATOR_VERSION, "split": self.split,
            "implementation_sha256": file_digest(Path(__file__)),
            "model": adapter.identity,
            "tasks": {task.id: task.identity for task in tasks},
            "batch_size": runtime.batch_size,
            "baselines": self._baselines,
            "runtime": {
                "device": runtime.device, "compute_dtype": runtime.compute_dtype,
                "seed": runtime.seed, "num_workers": runtime.num_workers,
                "rng_policy": "task_split_seeded_restore_v2",
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cpu_threads": torch.get_num_threads(),
                "cpu_interop_threads": torch.get_num_interop_threads(),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "numerical_environment": numerical_environment(runtime.device),
            },
        }

    @property
    def split(self) -> str:
        return self._split

    def with_baselines(self, baselines: Mapping[str, float] | None) -> "Evaluator":
        return Evaluator(self.adapter, tuple(self.tasks.values()), self.runtime, baselines)

    def evaluate(self, weights: Mapping[str, torch.Tensor], task_ids: Sequence[str] | None = None) -> EvaluationResult:
        requested = self.task_ids if task_ids is None else tuple(task_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Evaluation task list must be nonempty and unique")
        unknown = set(requested) - set(self.tasks)
        if unknown:
            raise ValueError(f"Unknown evaluation tasks: {sorted(unknown)}")
        self.adapter.load_weights(weights)
        per_task = {}
        with torch.no_grad():
            for task_id in requested:
                task = self.tasks[task_id]
                correct, total = 0, 0
                with evaluation_rng(task, self.split, self.runtime):
                    for batch in task.loader:
                        if isinstance(batch, Mapping):
                            images, labels = batch["images"], batch["labels"]
                        else:
                            images, labels = batch[:2]
                        if not isinstance(labels, torch.Tensor) or labels.dtype not in (
                            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                        ):
                            raise ValueError(f"{task_id}: labels must be integer tensors")
                        labels = labels.to(self.runtime.device, dtype=torch.long)
                        logits = self.adapter.logits(images, task_id)
                        if logits.ndim != 2 or logits.shape[0] != labels.numel() or logits.shape[1] != len(task.class_names):
                            raise ValueError(f"{task_id}: logits shape does not match batch/classes")
                        if not torch.isfinite(logits).all():
                            raise ValueError(f"{task_id}: nonfinite model outputs")
                        if (labels < 0).any() or (labels >= logits.shape[1]).any():
                            raise ValueError(f"{task_id}: label outside class range")
                        predictions = logits.argmax(dim=1)
                        correct += int((predictions == labels.reshape(-1)).sum().item())
                        total += int(labels.numel())
                if total != task.size or total == 0:
                    raise ValueError(f"{task_id}: evaluated {total} images, expected {task.size}")
                top1 = correct / total
                normalized = None if self._baselines is None else top1 / self._baselines[task_id]
                per_task[task_id] = TaskMetrics(correct, total, top1, normalized)
        absolute = sum(value.top1 for value in per_task.values()) / len(per_task)
        normalized = None if self._baselines is None else sum(value.normalized_top1 for value in per_task.values()) / len(per_task)
        return EvaluationResult(self.split, per_task, absolute, normalized,
                                sum(value.total for value in per_task.values()))


def build_evaluator(prepared: PreparedExperiment, split: str, adapter=None,
                    populate_baselines: bool = True) -> Evaluator:
    if split not in ("validation", "test"):
        raise ValueError(f"Invalid split {split}")
    if adapter is None:
        from src.models.adapter import OpenClipAdapter
        adapter = OpenClipAdapter(prepared.spec, prepared.merge_inputs.parameter_order)
    tasks = []
    for task_spec in prepared.spec.tasks:
        task = inspect_task(task_spec, split, prepared.spec.data_root,
                            adapter.preprocess, prepared.spec.runtime)
        task.identity["head"] = adapter.head_identity(task.id, task.class_names, task_spec.class_names)
        tasks.append(task)
    evaluator = Evaluator(adapter, tasks, prepared.spec.runtime)
    if populate_baselines:
        from src.evaluation.normalization import get_baselines
        evaluator = evaluator.with_baselines(get_baselines(prepared, split, evaluator))
    return evaluator
