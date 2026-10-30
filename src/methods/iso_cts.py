"""Iso-CTS, matching the official iso-merging implementation.

Reference: danielm1405/iso-merging, d34010af8ba129ef4d17a91ca9943b413fe06f8c,
src/utils/iso.py (blob 1e02c7b455ece261b8433e95f449432add2c7ee3).
Python round and residual left projection are
intentional. Non-matrix deltas use the same ordered running mean as TSV-M.
"""
from dataclasses import dataclass
import math
import torch

from src.core.numerics import cached_merge, deltas, spectral_update, svd, validate_scale


@dataclass(frozen=True)
class IsoCtsParams:
    alpha: float = 1.0
    common_space_fraction: float = 0.8

    def __post_init__(self):
        validate_scale(self.alpha)
        if not math.isfinite(self.common_space_fraction) or not 0 <= self.common_space_fraction <= 1:
            raise ValueError("common_space_fraction must be in [0, 1]")


def merge_layer(inputs, name, params, context):
    shape = inputs.base.tensor(name).shape
    count = len(inputs.task_order)
    r = min(shape)
    common_rank = int(r * params.common_space_fraction)
    specific_width = round((r - common_rank) / count) * count
    common_rank = r - specific_width
    if common_rank < 0:
        raise ValueError("official Iso-CTS rounding gives a negative common rank for this shape/task count")
    specific_rank = (r - common_rank) // count
    # sum(generator) preserves the reference's left-to-right tensor addition.
    combined = sum(deltas(inputs, name, context))
    u, s, v = svd(combined)
    common_u, common_s, common_v = u[:, :common_rank], s[:common_rank], v[:common_rank, :]
    for index, delta in enumerate(deltas(inputs, name, context)):
        residual = delta - common_u @ common_u.T @ delta
        task_u, task_s, task_v = svd(residual)
        if index == 0:
            combined_u, combined_s, combined_v = torch.zeros_like(task_u), torch.zeros_like(task_s), torch.zeros_like(task_v)
        start, end = index * specific_rank, (index + 1) * specific_rank
        combined_u[:, start:end] = task_u[:, :specific_rank]
        combined_s[start:end] = task_s[:specific_rank]
        combined_v[start:end, :] = task_v[:specific_rank, :]
    start = count * specific_rank
    combined_u[:, start:start + common_rank] = common_u
    combined_s[start:start + common_rank] = common_s
    combined_v[start:start + common_rank, :] = common_v
    left_u, _, right_u = svd(combined_u)
    left_v, _, right_v = svd(combined_v)
    combined_u = left_u @ right_u
    combined_v = left_v @ right_v
    combined_s = torch.ones_like(combined_s) * combined_s.mean()
    merged = torch.linalg.multi_dot((combined_u, torch.diag(combined_s), combined_v))
    return merged, {"r": r, "common_rank": common_rank,
                    "ranks": dict.fromkeys(inputs.task_order, specific_rank),
                    "isotropic_singular_value": float(combined_s[0]) if r else 0.0}


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    return cached_merge(inputs, params, context, "iso_cts_official_v1",
                        lambda: spectral_update(inputs, context, "iso_cts", lambda name: merge_layer(inputs, name, params, context)))
