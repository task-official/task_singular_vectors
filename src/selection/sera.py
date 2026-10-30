"""SERA target-energy search with a reproducible local GP/EI policy.

Paper: arXiv:2609.24517v1, section 4.2. The paper does not provide author
code or all GP details. This implementation explicitly uses an RBF GP,
maximum-likelihood length scale, EI on a fixed 1025-point grid, observation
noise 1e-6 (standardized objectives), and first-grid-point tie breaking.
It is a versioned paper implementation, not a claim of author-code identity.
``max_trials`` counts energy proposals; every alpha evaluation still consumes
the single image budget, including interrupted validation attempts.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from src.core.types import MergeContext, MergeInputs, MergeOutput, MethodDefinition, SelectionResult, SelectionSpec, ValidationEvaluator
from src.selection.search import SelectionSession, ValidationBudgetExceeded, plain_parameters, select as select_grid


def expected_improvement(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Predict EI in the objective's original units without global RNG state."""
    center = float(y.mean())
    scale = max(float(y.std()), 1e-8)
    target = (y - center) / scale
    distances = (x[:, None] - x[None, :]) ** 2

    def factor(log_length):
        covariance = np.exp(-0.5 * distances / math.exp(2 * log_length))
        covariance.flat[::len(x) + 1] += 1e-6
        return cho_factor(covariance, lower=True, check_finite=False)

    def likelihood(log_length):
        chol = factor(log_length)
        alpha = cho_solve(chol, target, check_finite=False)
        return float(0.5 * target @ alpha + np.log(np.diag(chol[0])).sum())

    optimum = minimize_scalar(likelihood, bounds=(-4.6, 0.0), method="bounded",
                              options={"xatol": 1e-6})
    log_length = float(optimum.x)
    chol = factor(log_length)
    cross = np.exp(-0.5 * (grid[:, None] - x[None, :]) ** 2 / math.exp(2 * log_length))
    mean = center + scale * (cross @ cho_solve(chol, target, check_finite=False))
    variance = np.maximum(1.0 - np.sum(cross * cho_solve(chol, cross.T, check_finite=False).T, axis=1), 0)
    sigma = scale * np.sqrt(variance)
    improvement = mean - float(y.max())
    z = improvement / np.maximum(sigma, 1e-15)
    ei = improvement * ndtr(z) + sigma * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    ei[sigma <= 1e-15] = 0
    return np.maximum(ei, 0)


def select(inputs: MergeInputs, method: MethodDefinition, protocol: SelectionSpec,
           validation: ValidationEvaluator, context: MergeContext,
           base_params: Mapping[str, Any] | None = None, *, record: Callable | None = None,
           resume_history: Sequence[dict] = (), resume_output: MergeOutput | None = None) -> SelectionResult:
    if protocol.mode != "bayes":
        return select_grid(inputs, method, protocol, validation, context, base_params,
                           record=record, resume_history=resume_history, resume_output=resume_output)
    session = SelectionSession(validation, protocol, record, resume_history, resume_output)
    low, high = protocol.energy_bounds
    if not 0 < low < high < 1:
        raise ValueError("SERA energy_bounds must satisfy 0 < low < high < 1")
    if isinstance(protocol.max_trials, bool) or not isinstance(protocol.max_trials, int) or protocol.max_trials < 1 or not protocol.initial_energies:
        raise ValueError("SERA requires positive max_trials and initial energies")
    if not math.isfinite(protocol.ei_tolerance) or protocol.ei_tolerance < 0:
        raise ValueError("ei_tolerance must be nonnegative")
    if any(not low <= e <= high for e in protocol.initial_energies):
        raise ValueError("Initial energies must lie within energy_bounds")
    if not protocol.candidates:
        raise ValueError("SERA requires at least one declared parameter candidate")
    if any("energy_threshold" in c for c in protocol.candidates):
        raise ValueError("Bayesian search owns energy_threshold; use grid mode for explicit energies")
    observations = [(float(e["energy"]), float(e["score"])) for e in session.history
                    if e.get("event") == "energy_observation"]
    proposed = [float(e["energy"]) for e in session.history if e.get("event") == "energy_proposal"]
    grid = np.linspace(low, high, 1025)
    while len(observations) < protocol.max_trials:
        observed_energies = [x for x, _ in observations]
        pending = [e for e in proposed if not any(abs(e - x) < 1e-12 for x in observed_energies)]
        initial = [float(e) for e in protocol.initial_energies
                   if not any(abs(e - x) < 1e-12 for x in observed_energies)]
        if pending:
            energy = pending[0]
        elif initial:
            energy = initial[0]
        else:
            x = np.asarray(observed_energies)
            y = np.asarray([v for _, v in observations])
            ei = expected_improvement(x, y, grid)
            for value in x:
                ei[np.abs(grid - value) < 1e-8] = -np.inf
            index = int(np.argmax(ei))
            if not math.isfinite(float(ei[index])):
                break
            span = float(np.ptp(y))
            if span > 0 and float(ei[index]) / span < protocol.ei_tolerance:
                session.append({"event": "selection_stop", "reason": "normalized_expected_improvement",
                                "normalized_ei": float(ei[index]) / span})
                break
            energy = float(grid[index])
        if energy not in proposed:
            session.append({"event": "energy_proposal", "energy": energy,
                            "policy": "sera_gp_rbf_ei_grid_v2_shared_objective", "seed": protocol.seed})
            proposed.append(energy)
        scores = []
        try:
            for override in protocol.candidates:
                params = method.parameters({**dict(base_params or {}), **dict(override),
                                            "energy_threshold": energy})
                values = plain_parameters(params)
                metrics = session.cached(values)
                if metrics is None:
                    session.ensure_budget()
                    output = method.merge(inputs, params, context, None)
                    metrics = session.evaluate(output, values)
                scores.append(session.score(metrics))
        except ValidationBudgetExceeded:
            if session.best_output is None:
                raise
            session.append({"event": "selection_stop", "reason": "validation_budget"})
            break
        score = max(scores)
        session.append({"event": "energy_observation", "energy": energy, "score": score})
        observations.append((energy, score))
    return session.result()
