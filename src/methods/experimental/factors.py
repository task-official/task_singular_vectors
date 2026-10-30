"""Layer-local factors shared by the research variants."""
import torch
from src.core.numerics import deltas, layer_trace, reconstruction_trace, timed


def decompose(inputs, name, context):
    # Retain factors for one layer on CPU, rather than all model/task deltas.
    return [tuple(t.cpu() for t in torch.linalg.svd(delta, full_matrices=False))
            for delta in deltas(inputs, name, context)]


def on_device(factors, context):
    return tuple([f[i].to(context.device) for f in factors] for i in range(3))


def orthogonalized_merge(
    u_list, s_list, v_list, ranks, device, return_reconstructions=False
):
    total_rank = sum(ranks)
    if total_rank == 0:
        return (None, None, None, None, None, None) if return_reconstructions else None

    sum_u = torch.zeros(
        u_list[0].shape[0],
        total_rank,
        device=device,
        dtype=u_list[0].dtype,
    )
    sum_s = torch.zeros(total_rank, device=device, dtype=s_list[0].dtype)
    sum_v = torch.zeros(
        total_rank,
        v_list[0].shape[1],
        device=device,
        dtype=v_list[0].dtype,
    )

    offset = 0
    for u, s, v, rank in zip(u_list, s_list, v_list, ranks):
        if rank == 0:
            continue
        next_offset = offset + rank
        sum_u[:, offset:next_offset] = u[:, :rank]
        sum_s[offset:next_offset] = s[:rank]
        sum_v[offset:next_offset, :] = v[:rank, :]
        offset = next_offset

    u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
    merged = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))
    if not return_reconstructions:
        return merged

    return merged, u_u, v_u, sum_s, u_v, v_v


def merge_factors(inputs, name, factors, ranks, context):
    u, s, v = on_device(factors, context)
    full = context.runtime.trace == "full"
    observed = orthogonalized_merge(u, s, v, ranks, context.device, return_reconstructions=full)
    merged = observed[0] if full else observed
    if merged is None:
        merged = torch.zeros_like(inputs.base.tensor(name), device=context.device)
    trace = layer_trace(inputs.task_order, [f[1] for f in factors], ranks)
    if full and sum(ranks):
        with timed(context, "trace_seconds"):
            _, u_u, v_u, singular, u_v, v_v = observed
            left, right = u_u @ v_u, u_v @ v_v
            trace["reconstruction"] = reconstruction_trace(inputs, name, left, singular, right, ranks, context)
    return merged, trace
