"""Task Arithmetic and averaging with the original global trim semantics.

The average reference is theta0 + sum(theta_t-theta0)/T, deliberately preserving
the operation sequence used by this repository, rather than substituting a mean.
"""
from dataclasses import dataclass
import torch

from src.core.numerics import cached_merge, timed, validate_fraction, validate_scale
from src.core.types import MergeOutput


@dataclass(frozen=True)
class AverageParams:
    retention_fraction: float = 1.0

    def __post_init__(self):
        validate_fraction(self.retention_fraction)


@dataclass(frozen=True)
class TaskArithmeticParams:
    alpha: float = 1.0
    retention_fraction: float = 1.0

    def __post_init__(self):
        validate_scale(self.alpha)
        validate_fraction(self.retention_fraction)


@dataclass(frozen=True)
class ZeroShotParams:
    pass


@torch.no_grad()
def merge_task_arithmetic(inputs, params, context, fitted_state=None):
    from .ties import trimmed_sum
    return cached_merge(inputs, params, context, "task_arithmetic", lambda: trimmed_sum(inputs, params.retention_fraction, context))


@torch.no_grad()
def merge_average(inputs, params, context, fitted_state=None):
    result = merge_task_arithmetic(inputs, TaskArithmeticParams(1 / len(inputs.task_order), params.retention_fraction), context)
    if result.trace:
        result.trace["method"] = "weight_average"
    return result


@torch.no_grad()
def merge_zeroshot(inputs, params, context, fitted_state=None):
    with timed(context, "merge_total_seconds"):
        return MergeOutput({key: inputs.base.tensor(key).clone().cpu() for key in inputs.parameter_order}, {})
