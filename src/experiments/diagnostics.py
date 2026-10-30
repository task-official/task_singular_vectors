"""Read actual merge traces; never recompute or select a model here."""
from __future__ import annotations

from typing import Any, Mapping


def analyze_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    if not trace:
        return {"status": "not_collected"}
    # All algorithm-specific quantities must come from the kernel that built
    # the evaluated weights. No alternate merge path is imported here.
    import torch

    def summary(value):
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "values": "see_saved_trace"}
        if isinstance(value, Mapping):
            return {str(k): summary(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [summary(v) for v in value]
        return value

    return {"status": "collected", "observed": summary(trace)}
