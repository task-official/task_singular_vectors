"""TSV-M: numerical reference a6c97188f7aa0bb20753de4f2939be89dd64f1c3.

Keep the original economy-SVD-width zero padding and fixed per-task slots,
including slots belonging to zero-energy tasks. Removing these changes polar
factors and is a different algorithm, not a memory optimization.
"""
from dataclasses import dataclass
import torch

from src.core.numerics import cached_merge, deltas, layer_trace, reconstruction_trace, spectral_update, svd, timed, validate_scale


@dataclass(frozen=True)
class TsvmParams:
    alpha: float = 1.0

    def __post_init__(self):
        validate_scale(self.alpha)


def merge_layer(inputs, name, context):
    spectra = []
    for index, delta in enumerate(deltas(inputs, name, context)):
        u, s, v = svd(delta)
        if index == 0:
            sum_u, sum_s, sum_v = torch.zeros_like(u), torch.zeros_like(s), torch.zeros_like(v)
        rank = int(s.shape[0] * (1 / len(inputs.task_order)))
        start, end = index * rank, (index + 1) * rank
        sum_u[:, start:end] = u[:, :rank]
        sum_s[start:end] = s[:rank]
        sum_v[start:end, :] = v[:rank, :]
        spectra.append(s.detach().cpu())
    left_u, _, right_u = svd(sum_u)
    left_v, _, right_v = svd(sum_v)
    merged = torch.linalg.multi_dot((left_u, right_u, torch.diag(sum_s), left_v, right_v))
    ranks = [rank] * len(spectra)
    trace = layer_trace(inputs.task_order, spectra, ranks)
    if context.runtime.trace == "full":
        with timed(context, "trace_seconds"):
            before = torch.linalg.multi_dot((sum_u, torch.diag(sum_s), sum_v))
            norm = float(before.norm())
            trace["procrustes_deformation"] = float((merged-before).norm())/norm if norm else None
            trace["reconstruction"] = reconstruction_trace(inputs, name, left_u @ right_u, sum_s,
                                                             left_v @ right_v, ranks, context)
    return merged, trace


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    return cached_merge(inputs, params, context, "tsvm_upstream_a6c9718",
                        lambda: spectral_update(inputs, context, "tsvm", lambda name: merge_layer(inputs, name, context)))
