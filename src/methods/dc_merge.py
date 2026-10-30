"""DC-Merge Full Fine-tuning reference (no LoRA singular-value smoothing).

Source: Tobeginwith/DC-Merge, 089af77a378562272461e0f041217ec440a4a230,
vision_fft_merge/src/utils/DC_utils.py (blob ab3915adb2c0ed12e73cb00821c6ddb0e91a82aa).
Cover-space ties intentionally differs from whole-model TIES.
"""
from dataclasses import dataclass
import torch

from src.core.numerics import cached_merge, deltas, layer_trace, spectral_update, svd, validate_fraction, validate_scale


@dataclass(frozen=True)
class DcMergeParams:
    alpha: float = 1.0
    retention_fraction: float = 0.001

    def __post_init__(self):
        validate_scale(self.alpha)
        validate_fraction(self.retention_fraction)


def aggregate_cover_space(matrices, retention_fraction):
    """Official DC topk threshold and ties_small zero-sum behavior."""
    validate_fraction(retention_fraction)
    filtered = []
    for matrix in matrices:
        magnitudes = matrix.abs().reshape(-1)
        k = max(1, int(retention_fraction * magnitudes.numel()))
        threshold = torch.topk(magnitudes, k).values.min()
        filtered.append(matrix * (matrix.abs() >= threshold))
    stacked = torch.stack(filtered)
    summed = stacked.sum(dim=0)
    keep = torch.sign(stacked) == torch.sign(summed).unsqueeze(0)
    count = keep.sum(dim=0)
    result = (stacked * keep).sum(dim=0) / count.clamp(min=1)
    return torch.where(summed == 0, torch.zeros_like(result), result)


def merge_layer(inputs, name, params, context):
    spectra, truncated = [], []
    for index, delta in enumerate(deltas(inputs, name, context)):
        u, s, v = svd(delta)
        if index == 0:
            sum_u, sum_v = torch.zeros_like(u), torch.zeros_like(v)
        rank = int(s.numel() * (1 / len(inputs.task_order)))
        start, end = index * rank, (index + 1) * rank
        sum_u[:, start:end] = u[:, :rank]
        sum_v[start:end, :] = v[:rank, :]
        truncated.append((u[:, :rank] @ torch.diag(s[:rank]) @ v[:rank, :]).cpu())
        spectra.append(s.cpu())
    left_u, _, right_u = svd(sum_u)
    left_v, _, right_v = svd(sum_v)
    cover_u, cover_v = left_u @ right_u, left_v @ right_v
    projected = [torch.linalg.multi_dot((cover_u.T, delta.to(context.device), cover_v.T)) for delta in truncated]
    combined = aggregate_cover_space(projected, params.retention_fraction)
    mask = torch.zeros_like(combined)
    block = mask.shape[0] // len(inputs.task_order)
    for i in range(len(inputs.task_order)):
        mask[i * block:(i + 1) * block, i * block:(i + 1) * block] = 1
    merged = torch.linalg.multi_dot((cover_u, combined * mask, cover_v))
    return merged, layer_trace(inputs.task_order, spectra, [rank] * len(spectra))


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    return cached_merge(inputs, params, context, "dc_merge_fft_official_v1",
                        lambda: spectral_update(inputs, context, "dc_merge", lambda name: merge_layer(inputs, name, params, context)))
