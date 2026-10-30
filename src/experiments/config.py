"""Resolve Hydra's input once; kernels receive explicit typed values."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from src.core.types import ExperimentSpec, RuntimeSpec, SelectionSpec, TaskSpec


def _path(root: str, template: str, model: str, task: str = "") -> str:
    expanded = Path(template.format(model=model, task=task)).expanduser()
    return str(expanded if expanded.is_absolute() else Path(root).expanduser() / expanded)


def resolve_config(config: DictConfig | Mapping[str, Any]) -> ExperimentSpec:
    data = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    # Execution control is not part of an experiment's scientific identity.
    data.pop("resume_from", None)
    required = {"benchmark", "model", "inputs", "method", "protocol", "runtime"}
    if missing := required - set(data):
        raise ValueError(f"Missing experiment sections: {sorted(missing)}")
    benchmark, model, inputs = data["benchmark"], data["model"], data["inputs"]
    method, protocol = data["method"], data["protocol"]
    model_name = model["name"]
    ids = tuple(benchmark["task_ids"])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("benchmark.task_ids must be a nonempty ordered set")
    runtime = RuntimeSpec(**data["runtime"])
    if runtime.compute_dtype != "float32":
        raise ValueError("The reference implementation currently supports float32 computation only")
    if runtime.trace not in ("none", "summary", "full"):
        raise ValueError("trace must be none, summary or full")
    if runtime.batch_size <= 0 or runtime.block_size <= 0 or runtime.num_workers < 0:
        raise ValueError("Invalid runtime sizes")
    search = dict(protocol.get("selection", {}))
    if search.get("mode", "fixed") != "fixed":
        for name in ("seed", "max_validation_images"):
            if name not in search or search[name] is None:
                raise ValueError(f"Tuned selection requires explicit {name}")
    if "candidates" in search:
        search["candidates"] = tuple(dict(candidate) for candidate in search["candidates"])
    for name in ("energy_bounds", "initial_energies"):
        if name in search:
            search[name] = tuple(search[name])
    selection = SelectionSpec(**search)
    if selection.objective not in ("macro_normalized_top1", "first_task_top1"):
        raise ValueError(f"Unknown validation objective: {selection.objective}")
    if selection.mode not in ("fixed", "grid", "bayes"):
        raise ValueError(f"Unknown selection mode {selection.mode}")
    if not selection.candidates:
        raise ValueError("At least one parameter candidate is required")
    if selection.max_validation_images is not None and selection.max_validation_images <= 0:
        raise ValueError("Validation image budget must be positive")
    if selection.mode == "fixed" and len(selection.candidates) != 1:
        raise ValueError("Fixed selection requires exactly one candidate")
    root = str(inputs["checkpoint_root"])
    overrides = benchmark.get("tasks", {})
    tasks = []
    checkpoints = {}
    for task in ids:
        override = dict(overrides.get(task, {}))
        if unknown := set(override) - {"validation", "test", "head", "class_names", "checkpoint"}:
            raise ValueError(f"Unknown task settings for {task}: {sorted(unknown)}")
        default_splits = {
            split: {"adapter": "legacy", "name": task, "split": split,
                    "profile": inputs.get("data_profile", "legacy_local")}
            for split in ("validation", "test")
        }
        validation = dict(override.get("validation", default_splits["validation"]))
        test = dict(override.get("test", default_splits["test"]))
        for split, value in (("validation", validation), ("test", test)):
            if "split" in value and value["split"] != split:
                raise ValueError(f"Wrong split declaration for {task}/{split}")
            value["split"] = split
        head_template = override.get("head", inputs["head"])
        tasks.append(TaskSpec(task, validation, test,
                              _path(root, head_template, model_name, task),
                              tuple(override.get("class_names", ()))))
        template = override.get("checkpoint", inputs["finetuned"])
        checkpoints[task] = _path(root, template, model_name, task)
    return ExperimentSpec(
        model=model_name, tasks=tuple(tasks), checkpoint_paths=checkpoints,
        base_path=_path(root, inputs["base"], model_name), data_root=str(inputs["data_root"]),
        method=method["id"], params=dict(method.get("parameters", {})),
        selection=selection, runtime=runtime, protocol=protocol["name"],
        benchmark=benchmark["name"], source_status=inputs.get("source_status", "local_unverified"),
        pretrained_tag=model.get("pretrained_tag", "openai"), resolved_config=data,
    )
