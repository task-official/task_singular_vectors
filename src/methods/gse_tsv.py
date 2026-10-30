"""GSE-TSV is this fork's GlobalRawEnergyTopR, not normalized-energy selection."""
from dataclasses import dataclass
import math
import torch

from src.core.numerics import cached_merge, concatenate_factors, deltas, layer_trace, polar, reconstruction_trace, spectral_update, svd, timed, validate_scale


@dataclass(frozen=True)
class GseTsvParams:
    alpha: float = 1.0
    eps: float = 1e-12

    def __post_init__(self):
        validate_scale(self.alpha)
        if not math.isfinite(self.eps) or self.eps < 0:
            raise ValueError("eps must be finite and nonnegative")


def select_raw_energy_components(spectra, eps):
    candidates = []
    for task_index, s in enumerate(spectra):
        if s.square().sum() <= eps:
            continue
        candidates.extend((float(value), component, task_index)
                          for component, value in enumerate(s.square()))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    ranks = [0] * len(spectra)
    for _, _, task_index in candidates[:spectra[0].numel()]:
        ranks[task_index] += 1
    return ranks


def merge_layer(inputs, name, params, context):
    # Factors for ONE parameter only, offloaded immediately from the GPU.
    factors = [tuple(value.cpu() for value in svd(delta)) for delta in deltas(inputs, name, context)]
    spectra = [factor[1] for factor in factors]
    # CPU and CUDA reductions can straddle eps differently. Offload factors,
    # but preserve the original merge device for the actual selection rule.
    ranks = select_raw_energy_components([s.to(context.device) for s in spectra], params.eps)
    trace = layer_trace(inputs.task_order, spectra, ranks)
    trace["energy_decision_device"] = str(context.device)
    if not sum(ranks):
        merged = torch.zeros_like(inputs.base.tensor(name), device=context.device)
    else:
        u, s, v = (value.to(context.device) for value in concatenate_factors(factors, ranks))
        polar_u, polar_v = polar(u), polar(v)
        merged = (polar_u * s.unsqueeze(0)) @ polar_v
        if context.runtime.trace == "full":
            with timed(context, "trace_seconds"):
                trace["reconstruction"] = reconstruction_trace(inputs, name, polar_u, s, polar_v, ranks, context)
                before = (u * s.unsqueeze(0)) @ v
                norm = float(before.norm())
                trace["procrustes_deformation"] = float((merged - before).norm()) / norm if norm else None
    return merged, trace


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    return cached_merge(inputs, params, context, "gse_tsv_raw_energy_device_v2",
                        lambda: spectral_update(inputs, context, "gse_tsv", lambda name: merge_layer(inputs, name, params, context)))
