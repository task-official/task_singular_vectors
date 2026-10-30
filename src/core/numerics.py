"""Shared tensor operations; algorithm-specific selection rules stay explicit."""
from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
import platform
import random
import time
from typing import Callable, Mapping

import torch
import numpy as np

from .types import MergeContext, MergeInputs, MergeOutput, StateDict


def numerical_environment(device: str) -> dict:
    """Numerical runtime evidence shared by evaluation caches and run records."""
    selected = torch.device(device)
    if selected.type == "cuda":
        props = torch.cuda.get_device_properties(selected)
        hardware = {"type": "cuda", "name": props.name, "capability": [props.major, props.minor]}
    else:
        hardware = {"type": selected.type, "machine": platform.machine(), "processor": platform.processor()}
    return {
        "hardware": hardware,
        "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_runtime": torch.backends.cudnn.version(),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment": {key: os.environ.get(key) for key in
                        ("CUBLAS_WORKSPACE_CONFIG", "MKL_CBWR", "MKL_THREADING_LAYER")},
    }


@contextmanager
def seeded_rng(seed: int, device: str = "cpu"):
    """Repeat evaluation randomness without consuming the caller's RNG streams."""
    selected = torch.device(device)
    devices = [selected.index if selected.index is not None else torch.cuda.current_device()] if selected.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def timed(context: MergeContext, name: str):
    """Separate observed trace work from kernel time, including CUDA completion."""
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        context.timings[name] = context.timings.get(name, 0.0) + time.perf_counter() - start


def validate_scale(alpha: float) -> None:
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be a finite real number")


def validate_fraction(value: float, name: str = "retention_fraction") -> None:
    if not math.isfinite(float(value)) or not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1]")


def is_matrix(name: str, tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 and "text_projection" not in name


def svd(matrix: torch.Tensor):
    return torch.linalg.svd(matrix, full_matrices=False)


def polar(matrix: torch.Tensor) -> torch.Tensor:
    left, _, right = svd(matrix)
    return left @ right


def rank_for_energy(s: torch.Tensor, threshold: float, eps: float = 1e-12) -> int:
    validate_fraction(threshold, "energy_threshold")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    total = s.square().sum()
    if total <= eps or s.numel() == 0:
        return 0
    cumulative = torch.cumsum(s.square(), dim=0) / total
    target = torch.tensor(threshold, device=s.device, dtype=s.dtype)
    # A rounded cumulative total may be below 1 at the legal endpoint.
    return min(s.numel(), int(torch.searchsorted(cumulative, target).item()) + 1)


def deltas(inputs: MergeInputs, name: str, context: MergeContext):
    for task in inputs.task_order:
        value = inputs.delta(task, name)
        if not value.is_floating_point():
            raise TypeError(f"{name}: integer parameters are not task vectors")
        yield value.to(device=context.device, dtype=getattr(torch, context.runtime.compute_dtype))


def average_delta(inputs: MergeInputs, name: str, context: MergeContext) -> torch.Tensor:
    """Knuth running mean in task order, matching the TSV reference."""
    result = None
    for i, value in enumerate(deltas(inputs, name, context)):
        if result is None:
            result = value.clone()
        else:
            result += (value - result) / (i + 1)
    if result is None:
        raise ValueError("at least one task is required")
    return result


def compose_weights(inputs: MergeInputs, update: Mapping[str, torch.Tensor], alpha: float, context: MergeContext) -> StateDict:
    validate_scale(alpha)
    output = {}
    for name in inputs.parameter_order:
        base = inputs.base.tensor(name)
        if name not in update:
            output[name] = base.clone().cpu()
            continue
        base_device = base.to(context.device)
        delta = update[name].to(device=context.device, dtype=base.dtype)
        output[name] = (base_device + alpha * delta).cpu()
    return output


def clone_trace(value):
    """Keep observed arrays on CPU and isolate observers from cached trace state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: clone_trace(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_trace(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_trace(item) for item in value)
    return deepcopy(value)


def prepared_update(inputs: MergeInputs, params, context: MergeContext, method: str,
                    build: Callable, fitted_state=None):
    """Retain only ONE prepared update, reusable across alpha candidates.

    This is an in-memory numerical cache, never a model/dataset identity cache.
    None/summary observations share a cache. Full capture has its own cache key
    so requesting heavier observations never returns an incomplete old trace;
    the numerical merge kernel and its operation order stay unchanged.
    """
    if not inputs.task_order or len(set(inputs.task_order)) != len(inputs.task_order):
        raise ValueError("task_order must be nonempty and unique")
    if context.runtime.compute_dtype not in ("float32", "float64"):
        raise ValueError("Only explicit float32/float64 numerical kernels are supported")
    if context.runtime.trace not in ("none", "summary", "full"):
        raise ValueError("trace must be 'none', 'summary' or 'full'")
    values = asdict(params)
    alpha = float(values.pop("alpha", 1.0))
    validate_scale(alpha)
    key = (inputs.identity, inputs.task_order, inputs.parameter_order, method,
           context.runtime.device, context.runtime.compute_dtype, context.runtime.seed,
           context.runtime.trace == "full",
           json.dumps(values, sort_keys=True), json.dumps(fitted_state, sort_keys=True))
    cached = context.cache.get("prepared_update")
    if cached is None or cached[0] != key:
        update, trace = build()
        # Do not retain GPU intermediates or all prior candidate updates.
        update = {name: value.detach().cpu() for name, value in update.items()}
        trace = clone_trace(trace)
        context.cache["prepared_update"] = (key, update, trace)
    else:
        _, update, trace = cached
    return update, trace


def cached_merge(inputs: MergeInputs, params, context: MergeContext, method: str,
                 build: Callable, fitted_state=None) -> MergeOutput:
    with timed(context, "merge_total_seconds"):
        update, trace = prepared_update(inputs, params, context, method, build, fitted_state)
        weights = compose_weights(inputs, update, float(getattr(params, "alpha", 1.0)), context)
        return MergeOutput(weights, clone_trace(trace) if context.runtime.trace != "none" else {})


def spectral_update(inputs: MergeInputs, context: MergeContext, method: str, layer_builder):
    update, layers = {}, {}
    for name in inputs.parameter_order:
        base = inputs.base.tensor(name)
        if not base.is_floating_point():
            continue
        if is_matrix(name, base):
            merged, layers[name] = layer_builder(name)
            update[name] = merged.detach().cpu()
            del merged
        else:
            update[name] = average_delta(inputs, name, context).detach().cpu()
    return update, {"method": method, "layers": layers}


def concatenate_factors(factors, ranks, *, padded_width=None):
    first_u, first_s, first_v = factors[0]
    width = sum(ranks) if padded_width is None else padded_width
    u = first_u.new_zeros((first_u.shape[0], width))
    s = first_s.new_zeros(width)
    v = first_v.new_zeros((width, first_v.shape[1]))
    offset = 0
    for (ui, si, vi), rank in zip(factors, ranks):
        if rank < 0 or rank > si.numel():
            raise ValueError("selected rank is outside SVD bounds")
        u[:, offset:offset + rank] = ui[:, :rank]
        s[offset:offset + rank] = si[:rank]
        v[offset:offset + rank, :] = vi[:rank, :]
        offset += rank
    return u, s, v


def layer_trace(tasks, singular_values, ranks):
    energy, totals, selected = {}, {}, {}
    for task, s, rank in zip(tasks, singular_values, ranks):
        denominator = float(s.square().sum().item())
        numerator = float(s[:rank].square().sum().item())
        energy[task] = numerator / denominator if denominator else 0.0
        totals[task], selected[task] = denominator, numerator
    return {"r": int(singular_values[0].numel()),
            "ranks": dict(zip(tasks, map(int, ranks))),
            "singular_values": {task: s.tolist() for task, s in zip(tasks, singular_values)},
            "total_rank": int(sum(ranks)), "selected_energy": energy,
            "total_energy": totals, "selected_energy_sum": selected}


def reconstruction_trace(inputs, name, left, singular_values, right, ranks, context):
    """Observe actual post-polar factors; never perform another merge or SVD."""
    offset, result = 0, {}
    for task, rank in zip(inputs.task_order, ranks):
        original = inputs.delta(task, name).to(device=context.device, dtype=left.dtype)
        reconstruction = torch.linalg.multi_dot((left[:, offset:offset + rank],
                                                  torch.diag(singular_values[offset:offset + rank]),
                                                  right[offset:offset + rank, :]))
        squared_error = float((original - reconstruction).square().sum())
        squared_norm = float(original.square().sum())
        result[task] = {"squared_error": squared_error, "squared_norm": squared_norm,
                        "score": 1 - squared_error / squared_norm if squared_norm else None}
        offset += rank
    return result
