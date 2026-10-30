"""Historical delta-only APIs backed by the same canonical numerical kernels."""
from __future__ import annotations

import hashlib

import torch

from src.core.types import MergeContext, MergeInputs, RuntimeSpec


class _DeltaSource:
    def __init__(self, values, identity, zero=False):
        self.values, self.identity, self.zero = values, identity, zero

    def keys(self):
        return tuple(sorted(self.values))

    def tensor(self, name):
        value = self.values[name].detach().cpu()
        return torch.zeros_like(value) if self.zero else value


def delta_inputs(task_vectors, config):
    tasks = tuple(config.DATASETS)
    if len(tasks) != len(task_vectors) or not tasks:
        raise ValueError("Task vectors and configured datasets must match")
    # This adapter has a fresh context on each call, so object identities cannot
    # leak into persisted experiment provenance or another call's cache.
    base = _DeltaSource(task_vectors[0].vector, "legacy-zero-base", zero=True)
    tuned = {task: _DeltaSource(vector.vector, f"legacy-delta-{id(vector.vector)}")
             for task, vector in zip(tasks, task_vectors)}
    names = base.keys()
    if any(source.keys() != names for source in tuned.values()):
        raise ValueError("Task-vector parameter names differ")
    identity = hashlib.sha256(str([(task, tuned[task].identity) for task in tasks]).encode()).hexdigest()
    return MergeInputs(base, tuned, tasks, names, identity)


def legacy_diagnostics(trace, tasks):
    """Translate observed quantities only; uncollected STI remains explicit."""
    layers = trace.get("layers", {})
    ranks, selection = {}, {}
    selected = dict.fromkeys(tasks, 0.0)
    totals = dict.fromkeys(tasks, 0.0)
    errors, norms = dict.fromkeys(tasks, 0.0), dict.fromkeys(tasks, 0.0)
    has_reconstruction = False
    for name, layer in layers.items():
        ranks[name] = {"r": layer["r"], "base_rank": layer["r"] // len(tasks),
                       "ranks": layer["ranks"], "total_rank": layer["total_rank"]}
        selection[name] = layer["selected_energy"]
        for task in tasks:
            selected[task] += layer.get("selected_energy_sum", {}).get(task, 0.0)
            totals[task] += layer.get("total_energy", {}).get(task, 0.0)
            if task in layer.get("reconstruction", {}):
                has_reconstruction = True
                item = layer["reconstruction"][task]
                errors[task] += item["squared_error"]
                norms[task] += item["squared_norm"]
    scores = {task: selected[task] / totals[task] if totals[task] else 0.0 for task in tasks}
    label = {"tsvm": "TSVM", "gse_tsv": "GlobalRawEnergyTopR"}.get(trace.get("method"), trace.get("method"))
    output = {"method": label, "datasets": list(tasks),
              "rank": {"per_layer": ranks, "total_per_layer": {name: item["total_rank"] for name, item in ranks.items()}},
              "selection_energy": {"per_layer": selection, "task_scores": scores,
                                   "avg_score": sum(scores.values()) / len(tasks),
                                   "min_score": min(scores.values())},
              "sti": {"status": "not_collected"},
              "observed_trace": trace}
    total_rank = sum(item["total_rank"] for item in ranks.values())
    total_width = sum(item["r"] for item in ranks.values())
    output["component_width"] = {"sum_selected_components": total_rank, "sum_max_rank": total_width,
                                 "normalized_width": total_rank / total_width if total_width else 0.0}
    output["procrustes"] = {"per_layer": {
        name: {"deformation_rate": layer.get("procrustes_deformation"),
               "width_ratio": layer["total_rank"] / layer["r"] if layer["r"] else 0.0}
        for name, layer in layers.items()}}
    if has_reconstruction:
        scores = {task: 1 - errors[task] / norms[task] if norms[task] else 0.0 for task in tasks}
        output["reconstruction"] = {"task_scores": scores,
                                    "avg_score": sum(scores.values()) / len(tasks),
                                    "min_score": min(scores.values())}
    return output


def _context(inputs, config, diagnostics):
    dtypes = {inputs.base.tensor(name).dtype for name in inputs.parameter_order
              if inputs.base.tensor(name).is_floating_point()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.float64):
        raise ValueError("Legacy SVD adapter requires uniform float32 or float64 tensors")
    dtype = str(next(iter(dtypes))).removeprefix("torch.")
    return MergeContext(RuntimeSpec(device=str(config.device), compute_dtype=dtype,
                                   trace="full" if diagnostics else "none",
                                   seed=int(getattr(config, "seed", torch.initial_seed()))))


def _merge_legacy(task_vectors, config, method_id, parameters=None, diagnostics=False):
    from src.experiments.registry import get_method
    inputs = delta_inputs(task_vectors, config)
    definition = get_method(method_id)
    params = definition.parameters(parameters or {})
    output = definition.merge(inputs, params, _context(inputs, config, diagnostics))
    if diagnostics:
        observed = legacy_diagnostics(output.trace, inputs.task_order)
        names = {"adaptive_tsv": "AdaptiveTSV", "eigen_tsv": "TSVM_2", "dummy": "dummy",
                 "subspace_tsv": "AdaptiveSubspaceTSV" if (parameters or {}).get("weighted") else "SubspaceTSV"}
        if method_id == "component_selection":
            names[method_id] = {"normalized_top_r": "GlobalEnergyTopR", "per_task_energy": "PerTask95FullConcat",
                                "full": "FullComponentFullConcat"}[params.selection]
        if method_id in names:
            observed["method"] = names[method_id]
        return output.weights, observed
    return output.weights


def compute_and_sum_svd_mem_reduction(task_vectors, config, return_diagnostics=False):
    return _merge_legacy(task_vectors, config, "tsvm", diagnostics=return_diagnostics)


def compute_adaptive_tsv_merge(task_vectors, config, return_diagnostics=False):
    params = {"energy_threshold": float(config.method.energy_threshold), "eps": float(config.method.eps)}
    return _merge_legacy(task_vectors, config, "adaptive_tsv", params, return_diagnostics)


def compute_adaptive_tsv_subspace(task_vectors, config, weighted=False, return_diagnostics=False):
    params = {"weighted": weighted, "energy_threshold": float(config.method.energy_threshold), "eps": float(config.method.eps)}
    return _merge_legacy(task_vectors, config, "subspace_tsv", params, return_diagnostics)


def compute_and_sum_svd_mem_reduction_2(task_vectors, config):
    return _merge_legacy(task_vectors, config, "eigen_tsv")


def compute_and_sum_svd_mem_reduction_dummy(task_vectors, config):
    return _merge_legacy(task_vectors, config, "dummy")


def compute_component_selection_tsv_merge(task_vectors, config, selection_mode, method_name,
                                         return_diagnostics=False):
    if selection_mode == "tsvm":
        return compute_and_sum_svd_mem_reduction(task_vectors, config, return_diagnostics)
    if selection_mode == "adaptive":
        return compute_adaptive_tsv_merge(task_vectors, config, return_diagnostics)
    if selection_mode == "global_raw_top_r":
        return _merge_legacy(task_vectors, config, "gse_tsv", {"eps": float(config.method.eps)}, return_diagnostics)
    modes = {"global_top_r": "normalized_top_r", "per_task_95": "per_task_energy", "full": "full"}
    if selection_mode not in modes:
        raise ValueError(f"Unknown component selection: {selection_mode}")
    params = {"selection": modes[selection_mode], "energy_threshold": float(getattr(config.method, "energy_threshold", .95)),
              "eps": float(config.method.eps)}
    return _merge_legacy(task_vectors, config, "component_selection", params, return_diagnostics)
