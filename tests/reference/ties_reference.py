"""Frozen TSV/TALL global TIES semantics, independent of production helpers."""
import torch


def ties_reference(rows, fraction):
    if fraction == 1:
        trimmed = rows.clone()
    else:
        kth = rows.shape[1] - int(rows.shape[1] * fraction)
        threshold = rows.abs().kthvalue(kth, dim=1, keepdim=True).values
        trimmed = rows * (rows.abs() >= threshold)
    signs = torch.sign(trimmed.sum(dim=0))
    signs[signs == 0] = torch.sign(signs.sum())
    selected = trimmed * torch.where(signs.unsqueeze(0) > 0, trimmed > 0, trimmed < 0)
    return selected.sum(dim=0) / (selected != 0).sum(dim=0).float().clamp(min=1)
