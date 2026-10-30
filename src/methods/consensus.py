"""Pure Consensus TA and TALL reconstruction kernels.

Mask lambdas are chosen by selection/consensus.py, never in these kernels.
The strict '>' mask boundary matches the original TALL code, and the order
base+sum-finetuned is preserved instead of algebraically rewriting it.
"""
from dataclasses import dataclass
import math
from typing import Mapping
import torch

from src.core.numerics import cached_merge, compose_weights, prepared_update, timed, validate_scale
from src.core.types import MergeOutput
from .arithmetic import TaskArithmeticParams
from .ties import trimmed_sum


@dataclass(frozen=True)
class ConsensusParams:
    alpha: float = 1.0
    min_tasks: int = 2
    mask_lambdas: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)

    def __post_init__(self):
        validate_scale(self.alpha)
        if isinstance(self.min_tasks, bool) or not isinstance(self.min_tasks, int) or self.min_tasks < 0:
            raise ValueError("min_tasks must be a nonnegative integer")
        if not self.mask_lambdas or any(not math.isfinite(x) or x < 0 for x in self.mask_lambdas):
            raise ValueError("mask_lambdas must be finite and nonnegative")


def _summed_update(inputs, context):
    return prepared_update(inputs, TaskArithmeticParams(), context, "consensus_sum",
                           lambda: trimmed_sum(inputs, 1.0, context))[0]


def task_mask(inputs, task, name, summed, mask_lambda):
    base = inputs.base.tensor(name)
    ft = inputs.finetuned[task].tensor(name)
    return (base - ft).abs() > (base + summed - ft).abs() * mask_lambda


@torch.no_grad()
def reconstruction(inputs, task, mask_lambda, context):
    if task not in inputs.task_order:
        raise ValueError(f"Unknown task {task}")
    if not math.isfinite(mask_lambda) or mask_lambda < 0:
        raise ValueError("mask_lambda must be finite and nonnegative")
    with timed(context, "merge_total_seconds"):
        summed = _summed_update(inputs, context)
        update = {name: value * task_mask(inputs, task, name, value, mask_lambda)
                  for name, value in summed.items()}
        return MergeOutput(compose_weights(inputs, update, 1.0, context),
                           {"method": "tall_reconstruction", "task": task, "mask_lambda": mask_lambda})


def build_mask_candidates(inputs, task, mask_lambdas, context):
    for mask_lambda in mask_lambdas:
        yield mask_lambda, reconstruction(inputs, task, mask_lambda, context)


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    if not isinstance(fitted_state, Mapping) or set(fitted_state) != set(inputs.task_order):
        raise ValueError("Consensus requires validation-selected lambdas for exactly the merged tasks")
    selected = {task: float(fitted_state[task]) for task in inputs.task_order}
    if any(not math.isfinite(x) or x < 0 for x in selected.values()):
        raise ValueError("selected mask lambdas must be finite and nonnegative")

    def build():
        summed = _summed_update(inputs, context)
        update, retained = {}, {}
        for name, value in summed.items():
            count = torch.zeros_like(value, dtype=torch.int32)
            for task in inputs.task_order:
                count += task_mask(inputs, task, name, value, selected[task])
            mask = count >= params.min_tasks
            update[name] = value * mask
            retained[name] = float(mask.float().mean())
        return update, {"method": "consensus_ta", "selected_lambdas": selected,
                        "min_tasks": params.min_tasks, "retained_fraction": retained}

    return cached_merge(inputs, params, context, "consensus_ta_tall_strict_v1", build, selected)
