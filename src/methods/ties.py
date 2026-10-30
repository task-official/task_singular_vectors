"""Whole-model TIES, preserving TSV/TALL's global kthvalue and zero-sign rules.

Trim thresholds are computed per task over ALL floating parameters. The elected
sign majority is also global. Blocks bound the task-by-parameter stack; neither
global operation is replaced by layerwise pruning. The source kthvalue rule can
retain one more entry than floor(P*fraction), and threshold ties retain all.
"""
from dataclasses import dataclass
import torch

from src.core.numerics import cached_merge, validate_fraction, validate_scale


@dataclass(frozen=True)
class TiesParams:
    alpha: float = 1.0
    retention_fraction: float = 0.2

    def __post_init__(self):
        validate_scale(self.alpha)
        validate_fraction(self.retention_fraction)


def parameter_layout(inputs):
    result = []
    for name in sorted(inputs.parameter_order):
        tensor = inputs.base.tensor(name)
        if tensor.is_floating_point():
            result.append((name, tensor.numel()))
    return result


def global_dtype(inputs):
    dtype = None
    for name, _ in parameter_layout(inputs):
        item = inputs.base.tensor(name).dtype
        dtype = item if dtype is None else torch.promote_types(dtype, item)
    return dtype


def compute_global_thresholds(inputs, retention_fraction):
    validate_fraction(retention_fraction)
    if retention_fraction == 1:
        return {task: None for task in inputs.task_order}
    layout = parameter_layout(inputs)
    thresholds = {}
    dtype = global_dtype(inputs)
    for task in inputs.task_order:
        # One full task vector on CPU, rather than all task checkpoints stacked.
        # Flattening in the reference promotes the entire state before subtraction.
        flat = torch.cat([(inputs.finetuned[task].tensor(name).to(dtype=dtype) -
                           inputs.base.tensor(name).to(dtype=dtype)).reshape(-1) for name, _ in layout])
        if flat.numel() == 0:
            raise ValueError("no floating parameters to trim")
        kth = flat.numel() - int(flat.numel() * retention_fraction)
        thresholds[task] = float(flat.abs().kthvalue(kth).values.item())
    return thresholds


def blocks(inputs, context):
    """Global contiguous blocks, crossing parameter boundaries when necessary.

    CPU reductions use a different accumulation kernel for narrow tails. Align
    internal boundaries to 64 elements and merge a final <32 tail into the
    previous block. This preserves the original flattened reduction's vector
    and scalar regions in the supported PyTorch CPU runtime. Small requested
    block sizes are rounded up; no block exceeds rounded_size+31 elements.
    """
    if context.runtime.block_size <= 0:
        raise ValueError("block_size must be positive")
    layout = parameter_layout(inputs)
    total = sum(size for _, size in layout)
    block_size = max(64, ((context.runtime.block_size + 63) // 64) * 64)
    start = 0
    while start < total:
        end = min(total, start + block_size)
        if 0 < total - end < 32:
            end = total
        segments, position = [], 0
        for name, size in layout:
            lo, hi = max(start, position), min(end, position + size)
            if lo < hi:
                segments.append((name, lo - position, hi - position))
            position += size
        yield start, end, segments
        start = end


def trimmed_block(inputs, segments, size, thresholds, dtype):
    bases = [(name, start, end, inputs.base.tensor(name).reshape(-1)[start:end].to(dtype=dtype))
             for name, start, end in segments]
    rows = []
    for task in inputs.task_order:
        delta = torch.empty(size, dtype=dtype)
        offset = 0
        for name, start, end, base in bases:
            tuned = inputs.finetuned[task].tensor(name).reshape(-1)[start:end].to(dtype=dtype)
            delta[offset:offset + end - start] = tuned - base
            offset += end - start
        threshold = thresholds[task]
        rows.append(delta if threshold is None else delta * (delta.abs() >= threshold))
    return torch.stack(rows)


def scatter_block(output, segments, values):
    offset = 0
    for name, start, end in segments:
        output[name].view(-1)[start:end] = values[offset:offset + end - start]
        offset += end - start


def trimmed_sum(inputs, retention_fraction, context):
    thresholds = compute_global_thresholds(inputs, retention_fraction)
    dtype = global_dtype(inputs)
    output = {name: torch.empty(inputs.base.tensor(name).shape, dtype=dtype) for name, _ in parameter_layout(inputs)}
    for start, end, segments in blocks(inputs, context):
        scatter_block(output, segments, trimmed_block(inputs, segments, end - start, thresholds, dtype).sum(dim=0))
    return output, {"method": "task_arithmetic", "thresholds": thresholds, "scope": "whole_model"}


def build_update(inputs, params, context):
    thresholds = compute_global_thresholds(inputs, params.retention_fraction)
    layout = parameter_layout(inputs)
    dtype = global_dtype(inputs)
    size = sum(count for _, count in layout)
    if size == 0:
        raise ValueError("no floating parameters to merge")
    # Preserve the source global float reduction, including the all-zero majority.
    signs = torch.empty(size, dtype=dtype)
    for start, end, segments in blocks(inputs, context):
        signs[start:end] = torch.sign(trimmed_block(inputs, segments, end - start, thresholds, dtype).sum(dim=0))
    majority = torch.sign(signs.sum())
    signs[signs == 0] = majority
    output = {name: torch.empty(inputs.base.tensor(name).shape, dtype=dtype) for name, _ in layout}
    for start, end, segments in blocks(inputs, context):
        values = trimmed_block(inputs, segments, end - start, thresholds, dtype)
        elected = signs[start:end]
        keep = torch.where(elected.unsqueeze(0) > 0, values > 0, values < 0)
        selected = values * keep
        counts = (selected != 0).sum(dim=0).float().clamp(min=1)
        scatter_block(output, segments, selected.sum(dim=0) / counts)
    return output, {"method": "ties", "scope": "whole_model", "thresholds": thresholds,
                    "zero_sign_rule": "global_majority", "majority_sign": float(majority)}


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    return cached_merge(inputs, params, context, "ties", lambda: build_update(inputs, params, context))
