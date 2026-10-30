"""Historical AdaptiveTSV with an explicit bounded energy endpoint."""
from dataclasses import dataclass
import math
import torch
from src.core.numerics import cached_merge, spectral_update, validate_scale, validate_fraction
from .factors import decompose, merge_factors
from .ranks import _adaptive_rank_budget


@dataclass(frozen=True)
class AdaptiveParams:
    alpha: float = 1.0
    energy_threshold: float = .95
    eps: float = 1e-12

    def __post_init__(self):
        validate_scale(self.alpha)
        validate_fraction(self.energy_threshold, "energy_threshold")
        if not math.isfinite(self.eps) or self.eps < 0:
            raise ValueError("eps must be finite and nonnegative")


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    def layer(name):
        factors = decompose(inputs, name, context)
        ranks = _adaptive_rank_budget([f[1].to(context.device) for f in factors], len(inputs.task_order),
                                      params.energy_threshold, params.eps)
        return merge_factors(inputs, name, factors, ranks, context)
    return cached_merge(inputs, params, context, "adaptive_tsv_endpoint_v2",
                        lambda: spectral_update(inputs, context, "adaptive_tsv", layer))
