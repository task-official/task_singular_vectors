"""Small explicit contracts shared by merging, selection and evaluation.

These types do not import Hydra, datasets, OpenCLIP or experiment runners.
Checkpoint readers return CPU tensors; numerical methods must never mutate them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

StateDict = dict[str, torch.Tensor]


class TensorSource(Protocol):
    identity: str

    def keys(self) -> tuple[str, ...]: ...
    def tensor(self, name: str) -> torch.Tensor: ...


@dataclass(frozen=True)
class MergeInputs:
    base: TensorSource
    finetuned: Mapping[str, TensorSource]
    task_order: tuple[str, ...]
    parameter_order: tuple[str, ...]
    identity: str

    def delta(self, task: str, name: str) -> torch.Tensor:
        return self.finetuned[task].tensor(name) - self.base.tensor(name)


@dataclass(frozen=True)
class RuntimeSpec:
    device: str = "cpu"
    compute_dtype: str = "float32"
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 0
    trace: str = "summary"
    block_size: int = 262144
    cache_dir: str = "cache"
    output_dir: str = "results/experiments"
    save_weights: bool = True

    def __post_init__(self):
        if not isinstance(self.save_weights, bool):
            raise ValueError("save_weights must be a boolean")


@dataclass
class MergeContext:
    runtime: RuntimeSpec
    cache: dict[Any, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return torch.device(self.runtime.device)


@dataclass
class MergeOutput:
    weights: StateDict
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskMetrics:
    correct: int
    total: int
    top1: float
    normalized_top1: float | None = None


@dataclass(frozen=True)
class EvaluationResult:
    split: str
    per_task: Mapping[str, TaskMetrics]
    avg_top1: float
    avg_normalized_top1: float | None
    evaluated_images: int


def validate_evaluation_result(result: EvaluationResult, split: str,
                               task_sizes: Mapping[str, int]) -> None:
    """Verify an evaluator's output before selecting, persisting or replaying it.

    Checking only ``split`` permits missing tasks, invalid counts and inconsistent
    aggregate scores to become durable experimental evidence.
    """
    if result.split != split or set(result.per_task) != set(task_sizes):
        raise ValueError(f"{split}: evaluator returned the wrong split or tasks")

    def integer(value):
        return isinstance(value, Integral) and not isinstance(value, bool)

    def finite(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    if not task_sizes or any(not integer(size) or size <= 0 for size in task_sizes.values()):
        raise ValueError("Evaluation requires positive integer task sizes")
    expected = sum(task_sizes.values())
    if not integer(result.evaluated_images) or result.evaluated_images != expected:
        raise ValueError(f"{split}: evaluation image count does not match declared task sizes")
    normalized = []
    for task, size in task_sizes.items():
        metrics = result.per_task[task]
        if (not integer(metrics.total) or metrics.total != size
                or not integer(metrics.correct) or not 0 <= metrics.correct <= size):
            raise ValueError(f"{split}/{task}: invalid correct/total counts")
        if not finite(metrics.top1) or not math.isclose(metrics.top1, metrics.correct / size, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{split}/{task}: accuracy does not match correct/total counts")
        value = metrics.normalized_top1
        if value is not None and (not finite(value) or value < 0):
            raise ValueError(f"{split}/{task}: normalized accuracy must be finite and nonnegative")
        normalized.append(value)
    macro = sum(result.per_task[task].top1 for task in task_sizes) / len(task_sizes)
    if not finite(result.avg_top1) or not math.isclose(result.avg_top1, macro, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{split}: average accuracy is not the task-wise mean")
    if all(value is None for value in normalized):
        if result.avg_normalized_top1 is not None:
            raise ValueError(f"{split}: normalized average has no per-task normalized accuracies")
    elif (any(value is None for value in normalized) or not finite(result.avg_normalized_top1)
          or not math.isclose(result.avg_normalized_top1, sum(normalized) / len(normalized), rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError(f"{split}: normalized average is not the task-wise mean")


class ValidationEvaluator(Protocol):
    split: str
    task_ids: tuple[str, ...]
    task_sizes: Mapping[str, int]

    def evaluate(self, weights: Mapping[str, torch.Tensor], task_ids: Sequence[str] | None = None) -> EvaluationResult: ...


@dataclass(frozen=True)
class SelectionSpec:
    mode: str = "fixed"
    candidates: tuple[dict[str, Any], ...] = field(default_factory=lambda: ({},))
    max_validation_images: int | None = None
    seed: int = 0
    energy_bounds: tuple[float, float] = (0.1, 0.99)
    initial_energies: tuple[float, ...] = (0.45, 0.65, 0.85)
    max_trials: int = 12
    ei_tolerance: float = 0.01
    objective: str = "macro_normalized_top1"


@dataclass
class SelectionResult:
    params: dict[str, Any]
    output: MergeOutput
    history: list[dict[str, Any]]
    fitted_state: Any = None


@dataclass(frozen=True)
class TaskSpec:
    id: str
    validation: Mapping[str, Any]
    test: Mapping[str, Any]
    head: str
    class_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentSpec:
    model: str
    tasks: tuple[TaskSpec, ...]
    checkpoint_paths: Mapping[str, str]
    base_path: str
    data_root: str
    method: str
    params: Mapping[str, Any]
    selection: SelectionSpec
    runtime: RuntimeSpec
    protocol: str = "common_fixed"
    benchmark: str = "custom"
    source_status: str = "local_unverified"
    pretrained_tag: str = "openai"
    resolved_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PreparedExperiment:
    spec: ExperimentSpec
    merge_inputs: MergeInputs
    identity: dict[str, Any]


MergeFunction = Callable[[MergeInputs, Any, MergeContext, Any], MergeOutput]


@dataclass(frozen=True)
class MethodDefinition:
    id: str
    merge: Callable[..., MergeOutput]
    params_type: type
    select: Callable[..., SelectionResult]
    implementation: str
    reference: str
    validate_inputs: Callable | None = None

    def parameters(self, values: Mapping[str, Any]) -> Any:
        return self.params_type(**dict(values))
