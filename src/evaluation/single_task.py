"""Individual checkpoints evaluated through the same runner and model adapter."""
from dataclasses import asdict, replace
from copy import deepcopy

from src.core.types import SelectionSpec
from src.experiments.runner import run_experiment


def run_individuals(spec, *, mode="standard", reconstructed=False, run=run_experiment,
                    method=None, parameters=None):
    if mode not in ("standard", "none"):
        raise ValueError("finetuning-mode must be standard or none")
    adapter = None

    def evaluator_factory(prepared, split, **kwargs):
        nonlocal adapter
        from src.evaluation.evaluator import build_evaluator
        if adapter is None:
            from src.models.adapter import OpenClipAdapter
            adapter = OpenClipAdapter(spec, prepared.merge_inputs.parameter_order)
        return build_evaluator(prepared, split, adapter=adapter)

    results = []
    for task in spec.tasks:
        selected_method = method or ("individual" if mode == "standard" else "zeroshot")
        params = dict(parameters) if parameters is not None else ({"reconstructed": reconstructed} if selected_method == "individual" else {})
        selection = SelectionSpec(seed=spec.selection.seed)
        config = deepcopy(dict(spec.resolved_config))
        config["benchmark"] = {**config.get("benchmark", {}), "name": f"{spec.benchmark}/{task.id}",
                               "task_ids": [task.id], "tasks": {key: value for key, value in config.get("benchmark", {}).get("tasks", {}).items() if key == task.id}}
        config["method"] = {"id": selected_method, "parameters": params}
        config["protocol"] = {"name": "individual_fixed", "selection": asdict(selection)}
        individual = replace(spec, tasks=(task,), checkpoint_paths={task.id: spec.checkpoint_paths[task.id]},
                             method=selected_method, params=params,
                             selection=selection, resolved_config=config, benchmark=f"{spec.benchmark}/{task.id}",
                             protocol="individual_fixed")
        results.append(run(individual, evaluator_factory=evaluator_factory))
    return results
