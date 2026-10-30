"""Shared subspace variants, preserving the historical matrix products."""
from dataclasses import dataclass
import torch
from src.core.numerics import cached_merge, layer_trace, spectral_update, timed
from .adaptive_tsv import AdaptiveParams
from .factors import decompose, on_device
from .ranks import _adaptive_rank_from_energy, _mean_adaptive_rank


@dataclass(frozen=True)
class SubspaceParams(AdaptiveParams):
    weighted: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.weighted, bool):
            raise ValueError("weighted must be a boolean")


def _top_eigenvectors(matrix, rank):
    _, eigenvectors = torch.linalg.eigh(matrix)
    return eigenvectors[:, -rank:]


def _adaptive_subspace_merge(
    u_list, s_list, v_list, ranks, device, eps, num_tasks, weighted, return_reconstructions=False
):
    common_rank = _mean_adaptive_rank(
        ranks, min(u_list[0].shape[0], v_list[0].shape[1])
    )
    if common_rank == 0:
        if return_reconstructions:
            zero_reconstructions = [
                torch.zeros(
                    u_list[0].shape[0],
                    v_list[0].shape[1],
                    device=device,
                    dtype=u_list[0].dtype,
                )
                for _ in ranks
            ]
            return None, zero_reconstructions, common_rank
        return None

    g_u = torch.zeros(
        u_list[0].shape[0],
        u_list[0].shape[0],
        device=device,
        dtype=u_list[0].dtype,
    )
    g_v = torch.zeros(
        v_list[0].shape[1],
        v_list[0].shape[1],
        device=device,
        dtype=v_list[0].dtype,
    )
    task_weight = 1 / num_tasks

    for u, s, v, rank in zip(u_list, s_list, v_list, ranks):
        if rank == 0:
            continue
        u_rank = u[:, :rank]
        v_rank = v[:rank, :].mT
        if weighted:
            energy = s[:rank].square()
            total_energy = energy.sum()
            if total_energy <= eps:
                continue
            g_u += task_weight * (
                (u_rank * energy.unsqueeze(0)) @ u_rank.mT
            ) / total_energy
            g_v += task_weight * (
                (v_rank * energy.unsqueeze(0)) @ v_rank.mT
            ) / total_energy
        else:
            g_u += task_weight * (u_rank @ u_rank.mT) / rank
            g_v += task_weight * (v_rank @ v_rank.mT) / rank

    p = _top_eigenvectors(g_u, common_rank)
    q = _top_eigenvectors(g_v, common_rank)
    c_merge = torch.zeros(
        common_rank,
        common_rank,
        device=device,
        dtype=u_list[0].dtype,
    )
    for u, s, v, rank in zip(u_list, s_list, v_list, ranks):
        if rank == 0:
            continue
        candidate = (u[:, :rank] * s[:rank].unsqueeze(0)) @ v[:rank, :]
        c_merge += p.mT @ candidate @ q

    merged = p @ c_merge @ q.mT
    if not return_reconstructions:
        return merged

    reconstructions = []
    for u, s, v, rank in zip(u_list, s_list, v_list, ranks):
        if rank == 0:
            reconstructions.append(
                torch.zeros(
                    u_list[0].shape[0],
                    v_list[0].shape[1],
                    device=device,
                    dtype=u_list[0].dtype,
                )
            )
            continue
        candidate = (u[:, :rank] * s[:rank].unsqueeze(0)) @ v[:rank, :]
        task_coefficients = p.mT @ candidate @ q
        reconstructions.append(p @ task_coefficients @ q.mT)

    return merged, reconstructions, common_rank


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    def layer(name):
        factors = decompose(inputs, name, context)
        u, s, v = on_device(factors, context)
        ranks = [_adaptive_rank_from_energy(si, params.energy_threshold, params.eps) for si in s]
        args = (u, s, v, ranks, context.device, params.eps, len(inputs.task_order), params.weighted)
        merged = _adaptive_subspace_merge(*args)
        if merged is None:
            merged = torch.zeros_like(inputs.base.tensor(name), device=context.device)
        trace = layer_trace(inputs.task_order, [f[1] for f in factors], ranks)
        trace["subspace_rank"] = _mean_adaptive_rank(ranks, min(merged.shape))
        return merged, trace
    return cached_merge(inputs, params, context, "subspace_tsv_endpoint_v2",
                        lambda: spectral_update(inputs, context, "subspace_tsv", layer))
