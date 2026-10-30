"""SERA paper implementation: arXiv:2609.24517v1, equations 10--24.

This is NOT an author-code reproduction or the old AdaptiveTSV algorithm.
Unspecified ties in the integer L2 projection are broken by task order. Zero
energy tasks receive raw demand 1 (the paper requires each allocated rank >=1).
A zero layer remains zero after reconstruction. Infeasible budgets (r<T) fail
explicitly rather than invent a rank-allocation rule outside the paper.
The default cover-space retention 0.001 follows the DC FFT reference; the paper
does not separately specify that constant. It is an explicit local parameter.
"""
from dataclasses import dataclass
import heapq
import math
import torch

from src.core.numerics import cached_merge, concatenate_factors, deltas, layer_trace, polar, rank_for_energy, spectral_update, svd, validate_fraction, validate_scale
from .dc_merge import aggregate_cover_space


@dataclass(frozen=True)
class SeraParams:
    alpha: float = 1.0
    energy_threshold: float = 0.95
    retention_fraction: float = 0.001
    eps: float = 1e-12

    def __post_init__(self):
        validate_scale(self.alpha)
        if not math.isfinite(self.energy_threshold) or not 0 < self.energy_threshold < 1:
            raise ValueError("SERA energy_threshold must be in (0, 1)")
        validate_fraction(self.retention_fraction)
        if not math.isfinite(self.eps) or self.eps < 0:
            raise ValueError("eps must be finite and nonnegative")


def validate_inputs(inputs, params):
    from src.core.numerics import is_matrix
    for name in inputs.parameter_order:
        tensor = inputs.base.tensor(name)
        if is_matrix(name, tensor) and min(tensor.shape) < len(inputs.task_order):
            raise ValueError(f"SERA rank budget is infeasible for {name}: r < tasks")


def project_integer_budget(demands, budget, max_rank):
    """Exact separable integer least-squares projection (Eq.14), task-order ties."""
    count = len(demands)
    if count == 0 or not count <= budget <= count * max_rank:
        raise ValueError("SERA rank budget is infeasible under ranks in [1, r]")
    total = sum(demands)
    if total <= 0:
        raise ValueError("rank demands must have positive sum")
    targets = [budget * float(value) / total for value in demands]
    ranks = [1] * count
    # Marginal objective increment for assigning one additional component.
    heap = [(3 - 2 * target, index) for index, target in enumerate(targets) if max_rank > 1]
    heapq.heapify(heap)
    for _ in range(budget - count):
        _, index = heapq.heappop(heap)
        ranks[index] += 1
        if ranks[index] < max_rank:
            heapq.heappush(heap, (2 * ranks[index] + 1 - 2 * targets[index], index))
    return ranks


def allocate_ranks(spectra, energy_threshold, eps=1e-12):
    max_rank, count = spectra[0].numel(), len(spectra)
    budget = count * (max_rank // count)
    raw = [max(1, rank_for_energy(s, energy_threshold, eps)) for s in spectra]
    return project_integer_budget(raw, budget, max_rank)


def merge_layer(inputs, name, params, context):
    factors = [tuple(value.cpu() for value in svd(delta)) for delta in deltas(inputs, name, context)]
    spectra = [factor[1] for factor in factors]
    ranks = allocate_ranks([s.to(context.device) for s in spectra], params.energy_threshold, params.eps)
    u, _, v = (value.to(context.device) for value in concatenate_factors(factors, ranks))
    cover_u, cover_v = polar(u), polar(v)
    projected, smoothed_values = [], {}
    for task, (task_u, task_s, task_v), rank in zip(inputs.task_order, factors, ranks):
        ui, si, vi = task_u[:, :rank].to(context.device), task_s[:rank].to(context.device), task_v[:rank, :].to(context.device)
        rms = torch.sqrt(si.square().sum() / rank)
        smoothed = ui @ torch.diag(torch.ones_like(si) * rms) @ vi
        projected.append(torch.linalg.multi_dot((cover_u.T, smoothed, cover_v.T)))
        smoothed_values[task] = float(rms)
    combined = aggregate_cover_space(projected, params.retention_fraction)
    mask = torch.zeros_like(combined)
    offset = 0
    for rank in ranks:
        mask[offset:offset + rank, offset:offset + rank] = 1
        offset += rank
    merged = torch.linalg.multi_dot((cover_u, combined * mask, cover_v))
    trace = layer_trace(inputs.task_order, spectra, ranks)
    trace.update({"smoothed_singular_value": smoothed_values,
                  "rank_decision_device": str(context.device),
                  "integer_projection": "exact_l2_task_order_ties",
                  "zero_energy_demand": 1})
    return merged, trace


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    return cached_merge(inputs, params, context, "sera_paper_2609.24517v1_device_v2",
                        lambda: spectral_update(inputs, context, "sera", lambda name: merge_layer(inputs, name, params, context)))
