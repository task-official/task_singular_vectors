"""Single-task rank ablation; retain the historical zero-filled SVD widths."""
from dataclasses import dataclass
import torch
from src.core.numerics import cached_merge, spectral_update, svd, validate_fraction, validate_scale


@dataclass(frozen=True)
class RankParams:
    alpha: float = 1.0
    rank_fraction: float = 1.0
    def __post_init__(self):
        validate_scale(self.alpha)
        validate_fraction(self.rank_fraction, "rank_fraction")


def validate_inputs(inputs, params):
    if len(inputs.task_order) != 1:
        raise ValueError("single_task_svd requires exactly one task")


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    validate_inputs(inputs, params)
    def layer(name):
        delta = inputs.delta(inputs.task_order[0], name).to(context.device)
        u, s, v = svd(delta)
        rank = int(s.numel() * params.rank_fraction)
        left, values, right = torch.zeros_like(u), torch.zeros_like(s), torch.zeros_like(v)
        left[:, :rank], values[:rank], right[:rank, :] = u[:, :rank], s[:rank], v[:rank, :]
        return torch.linalg.multi_dot((left, torch.diag(values), right)), {"rank": rank, "r": s.numel()}
    return cached_merge(inputs, params, context, "single_task_svd_v1",
                        lambda: spectral_update(inputs, context, "single_task_svd", layer))
