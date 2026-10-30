"""TALL task-mask fitting followed by shared Consensus TA selection."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from src.core.types import MergeContext, MergeInputs, MergeOutput, MethodDefinition, SelectionResult, SelectionSpec, ValidationEvaluator
from src.selection.search import SelectionSession, evaluate_candidates, plain_parameters


def select(inputs: MergeInputs, method: MethodDefinition, protocol: SelectionSpec,
           validation: ValidationEvaluator, context: MergeContext,
           base_params: Mapping[str, Any] | None = None, *, record: Callable | None = None,
           resume_history: Sequence[dict] = (), resume_output: MergeOutput | None = None) -> SelectionResult:
    from src.methods.consensus import reconstruction

    if protocol.mode == "bayes":
        raise ValueError("Consensus supports fixed/grid selection, not a hidden nested Bayesian search")
    session = SelectionSession(validation, protocol, record, resume_history, resume_output)
    params = method.parameters(dict(base_params or {}))
    base_values = plain_parameters(params)
    for override in protocol.candidates:
        if "mask_lambdas" in override and tuple(override["mask_lambdas"]) != tuple(params.mask_lambdas):
            raise ValueError("mask_lambdas is the declared fit grid; do not nest a second mask-grid search")
    fitted = {}
    for task in inputs.task_order:
        best_score, best_lambda = float("-inf"), None
        for mask_lambda in params.mask_lambdas:
            values = {"task": task, "mask_lambda": float(mask_lambda)}
            metrics = session.cached(values, "mask_fit", (task,))
            if metrics is None:
                session.ensure_budget((task,))
                output = reconstruction(inputs, task, float(mask_lambda), context)
                metrics = session.evaluate(output, values, stage="mask_fit", task_ids=(task,),
                                           track_best=False, objective=lambda m: m.per_task[task].top1)
            score = metrics.per_task[task].top1
            if score > best_score:
                best_score, best_lambda = score, float(mask_lambda)
        fitted[task] = best_lambda
    prior_fit = [e for e in session.history if e.get("event") == "masks_selected"]
    if prior_fit and prior_fit[-1]["lambdas"] != fitted:
        raise ValueError("Replayed mask choices disagree with the saved fit")
    if not prior_fit:
        session.append({"event": "masks_selected", "lambdas": fitted,
                        "mask_grid": list(params.mask_lambdas)})
    return evaluate_candidates(inputs, method, protocol, context, session, base_values, fitted)
