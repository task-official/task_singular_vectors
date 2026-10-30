"""Historical TSVM_2 eigen-based orthogonalization."""
from dataclasses import dataclass
import torch
from src.core.numerics import cached_merge, deltas, layer_trace, spectral_update, validate_scale


@dataclass(frozen=True)
class EigenParams:
    alpha: float = 1.0
    def __post_init__(self):
        validate_scale(self.alpha)


def merge_layer(inputs, name, context):
    spectra = []
    for index, delta in enumerate(deltas(inputs, name, context)):
        u, s, v = torch.linalg.svd(delta, full_matrices=False)
        if index == 0:
            sum_u, sum_s, sum_v = torch.zeros_like(u), torch.zeros_like(s), torch.zeros_like(v)
        rank = int(s.numel() * (1 / len(inputs.task_order)))
        start, end = index * rank, (index + 1) * rank
        sum_u[:, start:end], sum_s[start:end], sum_v[start:end, :] = u[:, :rank], s[:rank], v[:rank, :]
        spectra.append(s.cpu())
    sum_s, indices = torch.sort(sum_s, stable=True)

    sum_u = torch.index_select(sum_u, 1, indices)
    l_u, q_u = torch.linalg.eigh(sum_u.mT @ sum_u)
    u_orth = (
        q_u
        @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_u)) + 1e-12))
        @ q_u.mT
    )

    sum_v = torch.index_select(sum_v, 0, indices)

    l_v, q_v = torch.linalg.eigh(sum_v @ sum_v.mT)
    v_orth = (
        q_v
        @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_v)) + 1e-12))
        @ q_v.mT
    )

    merged = torch.linalg.multi_dot(  # bool_mask *
        (
            sum_u,
            u_orth,
            torch.diag(sum_s),
            v_orth,
            sum_v,
        )
    )
    return merged, layer_trace(inputs.task_order, spectra, [rank]*len(spectra))


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    return cached_merge(inputs, params, context, "eigen_tsv_v1",
                        lambda: spectral_update(inputs, context, "eigen_tsv", lambda name: merge_layer(inputs, name, context)))
