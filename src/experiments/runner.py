"""Resolve -> validation selection -> frozen-model test -> trace -> save."""
from __future__ import annotations

from dataclasses import asdict
import random
import resource
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.core.types import ExperimentSpec, MergeContext, PreparedExperiment, validate_evaluation_result
from src.core.checkpoints import tensor_fingerprint
from src.experiments.diagnostics import analyze_trace
from src.experiments.registry import get_method
from src.experiments.storage import RunStorage, jsonable, source_provenance, split_trace


def check_split_separation(validation, test):
    """Check dataset evidence before any candidate selection, without using scores."""
    if validation.split != "validation" or test.split != "test":
        raise ValueError("Evaluators are bound to the wrong splits")
    if tuple(validation.task_ids) != tuple(test.task_ids):
        raise ValueError("Validation/test task order differs")
    val_identity = getattr(validation, "identity", {}).get("tasks", {})
    test_identity = getattr(test, "identity", {}).get("tasks", {})
    for task in validation.task_ids:
        first = val_identity.get(task, {}).get("sample_digest")
        second = test_identity.get(task, {}).get("sample_digest")
        if first is not None and first == second:
            raise ValueError(f"{task}: validation and test contain identical observed datasets")
    validation_members = set()
    for task in getattr(validation, "tasks", {}).values():
        if getattr(task, "membership", None) is not None:
            validation_members.update(task.membership)
    for task in getattr(test, "tasks", {}).values():
        if getattr(task, "membership", None) is not None and validation_members.intersection(task.membership):
            raise ValueError(f"{task.id}: validation/test sample membership overlap")


def run_experiment(spec: ExperimentSpec, *, prepared: PreparedExperiment | None = None,
                   evaluator_factory: Callable | None = None, run_dir: str | Path | None = None,
                   resume: bool = False) -> dict:
    setup_started = time.perf_counter()
    # Heavy model/data imports occur only when a real evaluator is requested.
    from src.experiments.inputs import resolve_inputs
    if evaluator_factory is None:
        from src.evaluation.evaluator import build_evaluator
        evaluator_factory = build_evaluator
    random.seed(spec.runtime.seed)
    np.random.seed(spec.runtime.seed)
    torch.manual_seed(spec.runtime.seed)
    prepared = prepared or resolve_inputs(spec)
    if prepared.spec != spec:
        raise ValueError("Prepared experiment does not match the requested specification")
    method = get_method(spec.method)
    # Validate parameters before loading datasets/models or creating a run.
    parameters = method.parameters(spec.params)
    if method.validate_inputs is not None:
        method.validate_inputs(prepared.merge_inputs, parameters)
    for candidate in spec.selection.candidates:
        parameters = method.parameters({**dict(spec.params), **candidate})
        if method.validate_inputs is not None:
            method.validate_inputs(prepared.merge_inputs, parameters)
    validation = evaluator_factory(prepared, split="validation")
    validation_identity = jsonable(getattr(validation, "identity", {
        "split": "validation", "task_ids": validation.task_ids, "task_sizes": validation.task_sizes}))
    # Resolve both splits/heads before an expensive search. Test baselines are
    # constant per-task references, never candidate scores or selection inputs.
    test_kwargs = {"split": "test"}
    if hasattr(validation, "adapter"):
        test_kwargs["adapter"] = validation.adapter
    test_evaluator = evaluator_factory(prepared, **test_kwargs)
    check_split_separation(validation, test_evaluator)
    test_identity = jsonable(getattr(test_evaluator, "identity", {
        "split": "test", "task_ids": test_evaluator.task_ids,
        "task_sizes": test_evaluator.task_sizes}))
    provenance = source_provenance()
    config = jsonable(asdict(spec))
    identity = {"config": config, "inputs": prepared.identity, "validation": validation_identity,
                "test": test_identity,
                "method": {"id": method.id, "implementation": method.implementation,
                           "reference": method.reference}, "source": provenance}
    storage = RunStorage(spec.runtime.output_dir, identity, run_dir=run_dir, resume=resume)
    context = MergeContext(spec.runtime)
    setup_seconds = time.perf_counter() - setup_started
    started = time.perf_counter()
    if spec.runtime.device.startswith("cuda"):
        torch.cuda.synchronize(spec.runtime.device)
        torch.cuda.reset_peak_memory_stats(spec.runtime.device)
    try:
        selected = method.select(
            inputs=prepared.merge_inputs, method=method, protocol=spec.selection,
            validation=validation, context=context, base_params=spec.params,
            record=storage.observe, resume_history=storage.history,
            resume_output=storage.resume_output(),
        )
        frozen_weights_id = tensor_fingerprint(selected.output.weights)
        best_event = next(event for event in reversed(storage.history) if event.get("is_best"))
        if frozen_weights_id != best_event["weights_fingerprint"]:
            raise ValueError("Selected weights changed after validation")
        storage.mark("test", selected_params=selected.params)
        selection_seconds = time.perf_counter() - started
        # No selected parameters are changed after this boundary. The same
        # in-memory output (or its verified persisted copy on resume) is tested.
        before_test = time.perf_counter()
        if "test_identity" in storage.progress and storage.progress["test_identity"] != test_identity:
            raise ValueError("Test data/head/preprocessing changed during interrupted-run recovery")
        storage.mark("test", test_identity=test_identity)
        if "test_metrics" in storage.progress:
            from src.selection.search import metrics_from_record
            test_metrics = metrics_from_record(storage.progress["test_metrics"])
        else:
            test_metrics = test_evaluator.evaluate(selected.output.weights)
        if tensor_fingerprint(selected.output.weights) != frozen_weights_id:
            raise ValueError("Test evaluator mutated the selected weights")
        validate_evaluation_result(test_metrics, "test",
                                   {task: test_evaluator.task_sizes[task] for task in test_evaluator.task_ids})
        # Rejected outputs must never be cached: otherwise a resume could skip
        # the failed evaluator and publish metrics from a different model.
        if "test_metrics" not in storage.progress:
            storage.mark("test", test_metrics=asdict(test_metrics))
        test_seconds = time.perf_counter() - before_test
        before_diagnostics = time.perf_counter()
        diagnostics = analyze_trace(selected.output.trace)
        diagnostics, _ = split_trace(diagnostics)
        diagnostics_seconds = time.perf_counter() - before_diagnostics
        result = {
            "schema_version": 1, "status": "complete", "source_type": "local",
            "config": config, "input_identity": prepared.identity,
            "validation_identity": validation_identity,
            "test_identity": test_identity,
            "method": identity["method"], "selected_params": selected.params,
            "fitted_state": selected.fitted_state, "validation_history": selected.history,
            "test": asdict(test_metrics), "diagnostics": diagnostics,
            "provenance": provenance, "run_dir": str(storage.path),
            "performance": {
                "setup_seconds": setup_seconds,
                "selection_seconds": selection_seconds, "test_seconds": test_seconds,
                "diagnostics_seconds": diagnostics_seconds,
                "merge_total_seconds": context.timings.get("merge_total_seconds", 0.0),
                "trace_collection_seconds": context.timings.get("trace_seconds", 0.0),
                "merge_kernel_seconds": max(0.0, context.timings.get("merge_total_seconds", 0.0) - context.timings.get("trace_seconds", 0.0)),
                "resumed": resume,
                "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "rss_scope": "process lifetime high-water mark including setup; Linux ru_maxrss",
                "timing_scope": "current invocation only; setup includes input/model/both-split baseline preparation; selection includes merging, validation and durable journal; merge_kernel includes summary metadata/transfers but excludes measured full-trace collection; test evaluates only frozen selected weights; diagnostics reads trace only; peak GPU memory includes trace collection",
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(spec.runtime.device))
                if spec.runtime.device.startswith("cuda") else None,
            },
        }
        storage.finish(result, selected.output, save_weights=spec.runtime.save_weights)
        # Return exactly the published record, including the saved model digest.
        import json
        return json.loads((storage.path / "result.json").read_text())
    except BaseException as error:
        if not (storage.path / "result.json").exists():
            storage.mark(storage.progress.get("phase", "selection"), status="interrupted",
                         error_type=type(error).__name__, error=str(error))
        raise
    finally:
        storage.close()
