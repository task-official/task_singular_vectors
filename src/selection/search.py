"""Fixed/grid selection with one ledger for every validation evaluation.

An evaluation-start record reserves its full image cost before inference. A
crashed evaluation therefore cannot become free validation access on resume.
Only the runner receives a test evaluator; this module rejects other splits.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from typing import Any, Callable, Mapping, Sequence

from src.core.types import (
    EvaluationResult, MergeContext, MergeInputs, MergeOutput, MethodDefinition,
    SelectionResult, SelectionSpec, TaskMetrics, ValidationEvaluator,
    validate_evaluation_result,
)
from src.core.checkpoints import tensor_fingerprint


class ValidationBudgetExceeded(RuntimeError):
    """The next complete validation evaluation would exceed its image budget."""


def plain_parameters(params: Any) -> dict[str, Any]:
    return asdict(params) if is_dataclass(params) else dict(params)


def _key(stage: str, params: Mapping[str, Any], tasks: Sequence[str]) -> str:
    return json.dumps([stage, dict(params), list(tasks)], sort_keys=True, separators=(",", ":"))


def metrics_record(metrics: EvaluationResult) -> dict[str, Any]:
    return asdict(metrics)


def metrics_from_record(value: Mapping[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        split=value["split"],
        per_task={name: TaskMetrics(**item) for name, item in value["per_task"].items()},
        avg_top1=value["avg_top1"],
        avg_normalized_top1=value.get("avg_normalized_top1"),
        evaluated_images=value["evaluated_images"],
    )


class SelectionSession:
    """A validation capability, budget and ordered history for one selection.

    ``record`` is an optional durable observer owned by the runner. It gets a
    MergeOutput only when a new best *final shared* model has been observed.
    Task-specific mask reconstruction evaluations never replace that model.
    """

    def __init__(self, validation: ValidationEvaluator, protocol: SelectionSpec,
                 record: Callable | None = None, resume_history: Sequence[dict] = (),
                 resume_output: MergeOutput | None = None):
        if validation.split != "validation":
            raise ValueError("Selection requires a validation-only evaluator")
        if protocol.mode not in {"fixed", "grid", "bayes"}:
            raise ValueError(f"Unknown selection mode: {protocol.mode}")
        if protocol.mode != "fixed" and protocol.max_validation_images is None:
            raise ValueError("Tuned selection requires max_validation_images")
        if protocol.max_validation_images is not None and (
            isinstance(protocol.max_validation_images, bool)
            or not isinstance(protocol.max_validation_images, int)
            or protocol.max_validation_images <= 0
        ):
            raise ValueError("max_validation_images must be positive")
        if protocol.objective not in ("macro_normalized_top1", "first_task_top1"):
            raise ValueError(f"Unknown validation objective: {protocol.objective}")
        if not validation.task_ids or any(validation.task_sizes[t] <= 0 for t in validation.task_ids):
            raise ValueError("Validation tasks must be nonempty with positive sizes")
        self.validation = validation
        self.protocol = protocol
        self.record = record
        self.history = [dict(event) for event in resume_history]
        self.used_images = sum(int(e["expected_images"]) for e in self.history
                               if e.get("event") == "evaluation_start")
        self.completed = {e["key"]: e for e in self.history if e.get("event") == "evaluation_complete"}
        best = [e for e in self.history if e.get("event") == "evaluation_complete" and e.get("is_best")]
        self.best_score = float(best[-1]["score"]) if best else -math.inf
        self.best_params = dict(best[-1]["params"]) if best else None
        self.best_output = resume_output
        if best and resume_output is None:
            raise ValueError("Resume history contains a selected model but no saved weights")
        if not best and resume_output is not None:
            raise ValueError("Saved resume model has no corresponding validation event")

    def append(self, event: dict, output: MergeOutput | None = None) -> None:
        event = dict(event)
        event["sequence"] = len(self.history)
        if self.record is not None:
            self.record(event, output)
        self.history.append(event)

    def cached(self, params: Mapping[str, Any], stage: str = "candidate",
               task_ids: Sequence[str] | None = None) -> EvaluationResult | None:
        tasks = tuple(task_ids) if task_ids is not None else self.validation.task_ids
        event = self.completed.get(_key(stage, params, tasks))
        return metrics_from_record(event["metrics"]) if event else None

    def ensure_budget(self, task_ids: Sequence[str] | None = None) -> int:
        tasks = tuple(task_ids) if task_ids is not None else self.validation.task_ids
        expected = sum(int(self.validation.task_sizes[t]) for t in tasks)
        budget = self.protocol.max_validation_images
        if budget is not None and self.used_images + expected > budget:
            raise ValidationBudgetExceeded(
                f"Validation budget {budget} images: used {self.used_images}, next evaluation needs {expected}")
        return expected

    def evaluate(self, output: MergeOutput, params: Mapping[str, Any], *,
                 stage: str = "candidate", task_ids: Sequence[str] | None = None,
                 track_best: bool = True, objective: Callable[[EvaluationResult], float] | None = None) -> EvaluationResult:
        tasks = tuple(task_ids) if task_ids is not None else self.validation.task_ids
        if not tasks or len(tasks) != len(set(tasks)) or any(t not in self.validation.task_ids for t in tasks):
            raise ValueError("Evaluation tasks must be a nonempty unique validation subset")
        cached = self.cached(params, stage, tasks)
        if cached is not None:
            return cached
        expected = self.ensure_budget(tasks)
        key = _key(stage, params, tasks)
        self.append({"event": "evaluation_start", "key": key, "stage": stage,
                     "params": dict(params), "tasks": list(tasks), "expected_images": expected})
        self.used_images += expected
        weights_id = tensor_fingerprint(output.weights)
        metrics = self.validation.evaluate(output.weights, task_ids=tasks)
        if tensor_fingerprint(output.weights) != weights_id:
            raise ValueError("Validation evaluator mutated the candidate weights")
        validate_evaluation_result(metrics, "validation", {task: self.validation.task_sizes[task] for task in tasks})
        score = self.score(metrics, objective)
        # Strict > preserves the first declared candidate on exact ties.
        is_best = track_best and float(score) > self.best_score
        event = {"event": "evaluation_complete", "key": key, "stage": stage,
                 "params": dict(params), "tasks": list(tasks), "score": float(score),
                 "metrics": metrics_record(metrics), "is_best": is_best,
                 "weights_fingerprint": weights_id,
                 "used_validation_images": self.used_images}
        self.append(event, output if is_best else None)
        self.completed[key] = event
        if is_best:
            self.best_score, self.best_params, self.best_output = float(score), dict(params), output
        return metrics

    def score(self, metrics: EvaluationResult, objective=None) -> float:
        if objective is not None:
            score = objective(metrics)
        elif self.protocol.objective == "first_task_top1":
            score = metrics.per_task[self.validation.task_ids[0]].top1
        elif self.protocol.objective == "macro_normalized_top1":
            score = metrics.avg_normalized_top1
        else:
            raise ValueError(f"Unknown validation objective: {self.protocol.objective}")
        if score is None or not math.isfinite(float(score)):
            raise ValueError("Selection requires a finite normalized validation objective")
        return float(score)

    def result(self, fitted_state: Any = None) -> SelectionResult:
        if self.best_output is None or self.best_params is None:
            raise ValidationBudgetExceeded("No final model has completed validation within the budget")
        return SelectionResult(self.best_params, self.best_output, self.history, fitted_state)


def evaluate_candidates(inputs: MergeInputs, method: MethodDefinition,
                        protocol: SelectionSpec, context: MergeContext,
                        session: SelectionSession, base_params: Mapping[str, Any] | None = None,
                        fitted_state: Any = None) -> SelectionResult:
    if not protocol.candidates:
        raise ValueError("At least one parameter candidate is required")
    if protocol.mode == "fixed" and len(protocol.candidates) != 1:
        raise ValueError("Fixed selection requires exactly one candidate")
    for override in protocol.candidates:
        params = method.parameters({**dict(base_params or {}), **dict(override)})
        values = plain_parameters(params)
        if session.cached(values) is not None:
            continue
        try:
            session.ensure_budget()
            # We keep the very same output that validation observed.
            output = method.merge(inputs, params, context, fitted_state)
            session.evaluate(output, values)
        except ValidationBudgetExceeded:
            if session.best_output is None:
                raise
            session.append({"event": "selection_stop", "reason": "validation_budget"})
            break
    return session.result(fitted_state)


def select(inputs: MergeInputs, method: MethodDefinition, protocol: SelectionSpec,
           validation: ValidationEvaluator, context: MergeContext,
           base_params: Mapping[str, Any] | None = None, *, record: Callable | None = None,
           resume_history: Sequence[dict] = (), resume_output: MergeOutput | None = None) -> SelectionResult:
    if protocol.mode == "bayes":
        raise ValueError("This method has no Bayesian selection recipe")
    session = SelectionSession(validation, protocol, record, resume_history, resume_output)
    return evaluate_candidates(inputs, method, protocol, context, session, base_params)
