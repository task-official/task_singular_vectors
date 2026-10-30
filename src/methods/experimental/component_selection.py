"""Normalized-energy, per-task-energy and full-concatenation research methods."""
from dataclasses import dataclass
import torch
from src.core.numerics import cached_merge, concatenate_factors, layer_trace, polar, spectral_update, reconstruction_trace, timed
from .adaptive_tsv import AdaptiveParams
from .factors import decompose
from .ranks import _adaptive_rank_from_energy


@dataclass(frozen=True)
class ComponentParams(AdaptiveParams):
    selection: str = "normalized_top_r"

    def __post_init__(self):
        super().__post_init__()
        if self.selection not in ("normalized_top_r", "per_task_energy", "full"):
            raise ValueError("Unknown component selection rule")


def select_ranks(spectra, params):
    if params.selection == "per_task_energy":
        return [_adaptive_rank_from_energy(s, params.energy_threshold, params.eps) for s in spectra]
    if params.selection == "full":
        return [s.numel() if s.square().sum() > params.eps else 0 for s in spectra]
    candidates = []
    for task, s in enumerate(spectra):
        total = s.square().sum()
        if total > params.eps:
            candidates.extend((float(value), component, task)
                              for component, value in enumerate(s.square() / total))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    ranks = [0] * len(spectra)
    for _, _, task in candidates[:spectra[0].numel()]:
        ranks[task] += 1
    return ranks


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    def layer(name):
        factors = decompose(inputs, name, context)
        ranks = select_ranks([f[1].to(context.device) for f in factors], params)
        trace = layer_trace(inputs.task_order, [f[1] for f in factors], ranks)
        if not sum(ranks):
            return torch.zeros_like(inputs.base.tensor(name), device=context.device), trace
        u, s, v = (t.to(context.device) for t in concatenate_factors(factors, ranks))
        left, right = polar(u), polar(v)
        merged = (left * s.unsqueeze(0)) @ right
        if context.runtime.trace == "full":
            with timed(context, "trace_seconds"):
                before = (u * s.unsqueeze(0)) @ v
                norm = float(before.norm())
                trace["procrustes_deformation"] = float((merged-before).norm())/norm if norm > params.eps else None
                trace["reconstruction"] = reconstruction_trace(inputs, name, left, s, right, ranks, context)
        return merged, trace
    return cached_merge(inputs, params, context, "component_selection_endpoint_v2",
                        lambda: spectral_update(inputs, context, params.selection, layer))
