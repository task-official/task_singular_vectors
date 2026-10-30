"""Selection tests use synthetic tensors and no model/data downloads."""
from dataclasses import dataclass, replace
import unittest
from unittest.mock import patch

import torch

from src.core.checkpoints import make_inputs
from src.core.types import (
    EvaluationResult, MergeContext, MergeOutput, MethodDefinition, RuntimeSpec,
    SelectionSpec, TaskMetrics,
)
from src.selection import consensus, search, sera


@dataclass(frozen=True)
class Params:
    alpha: float = 1.0


class FakeValidation:
    split = "validation"
    task_ids = ("A", "B")
    task_sizes = {"A": 10, "B": 10}

    def __init__(self, peak=1.0, fail_call=None):
        self.peak, self.fail_call, self.calls = peak, fail_call, []

    def evaluate(self, weights, task_ids=None):
        tasks = self.task_ids if task_ids is None else tuple(task_ids)
        self.calls.append((weights, tasks))
        if self.fail_call == len(self.calls):
            raise RuntimeError("simulated interrupted validation")
        score = max(0.0, 1 - abs(float(weights["w"].item()) - self.peak))
        count = int(round(score * 10))
        per_task = {t: TaskMetrics(count, 10, count / 10, count / 10) for t in tasks}
        return EvaluationResult("validation", per_task, count / 10, count / 10, 10 * len(tasks))


def fake_merge(inputs, params, context, fitted_state=None):
    return MergeOutput({"w": torch.tensor(float(params.alpha))}, {"alpha": params.alpha})


def definition(params_type=Params, merge=fake_merge, selector=search.select):
    return MethodDefinition("fake", merge, params_type, selector, "test_v1", "synthetic")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.inputs = make_inputs({"w": torch.tensor(0.)}, {
            "A": {"w": torch.tensor(1.)}, "B": {"w": torch.tensor(2.)}})
        self.context = MergeContext(RuntimeSpec())

    def test_grid_keeps_exact_evaluated_output_and_first_tie(self):
        validation = FakeValidation()
        protocol = SelectionSpec("grid", ({"alpha": .5}, {"alpha": 1.}, {"alpha": 1.01}), 60)
        result = search.select(self.inputs, definition(), protocol, validation, self.context)
        self.assertEqual(result.params["alpha"], 1.)
        self.assertIs(result.output.weights, validation.calls[1][0])
        self.assertEqual(len(validation.calls), 3)

    def test_budget_stops_before_extra_forward(self):
        validation = FakeValidation()
        protocol = SelectionSpec("grid", ({"alpha": 0.}, {"alpha": 1.}, {"alpha": 2.}), 40)
        result = search.select(self.inputs, definition(), protocol, validation, self.context)
        self.assertEqual(result.params["alpha"], 1.)
        self.assertEqual(len(validation.calls), 2)
        self.assertEqual(sum(e["expected_images"] for e in result.history if e["event"] == "evaluation_start"), 40)

    def test_tuning_requires_budget_and_test_capability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_validation_images"):
            search.select(self.inputs, definition(), SelectionSpec(mode="grid"), FakeValidation(), self.context)
        validation = FakeValidation()
        validation.split = "test"
        with self.assertRaisesRegex(ValueError, "validation-only"):
            search.select(self.inputs, definition(), SelectionSpec(), validation, self.context)

    def test_interruption_costs_budget_and_resume_does_not_repeat_completed_trials(self):
        history, saved = [], []

        def observer(event, output):
            history.append(event)
            if output is not None:
                saved[:] = [output]

        protocol = SelectionSpec("grid", ({"alpha": 0.}, {"alpha": 1.}, {"alpha": 2.}), 80)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            search.select(self.inputs, definition(), protocol, FakeValidation(fail_call=2),
                          self.context, record=observer)
        validation = FakeValidation()
        result = search.select(self.inputs, definition(), protocol, validation, self.context,
                               resume_history=history, resume_output=saved[0])
        self.assertEqual(len(validation.calls), 2)
        self.assertEqual(result.params["alpha"], 1.)
        self.assertEqual(sum(e["expected_images"] for e in result.history if e["event"] == "evaluation_start"), 80)

    def test_consensus_records_mask_fit_before_shared_model_selection(self):
        from src.methods.consensus import ConsensusParams
        validation = FakeValidation(peak=.4)
        method = definition(ConsensusParams, fake_merge, consensus.select)
        protocol = SelectionSpec("grid", ({"alpha": 1.}, {"alpha": 2.}), 100)
        with patch("src.methods.consensus.reconstruction", side_effect=lambda i, t, lam, c: MergeOutput({"w": torch.tensor(lam)})):
            result = consensus.select(self.inputs, method, protocol, validation, self.context,
                                      {"mask_lambdas": (.2, .4, .6)})
        self.assertEqual(result.fitted_state, {"A": .4, "B": .4})
        starts = [e for e in result.history if e["event"] == "evaluation_start"]
        self.assertEqual(sum(e["expected_images"] for e in starts), 100)
        self.assertEqual(sum(e["stage"] == "mask_fit" for e in starts), 6)
        self.assertEqual(sum(e["stage"] == "candidate" for e in starts), 2)

    def test_sera_search_is_bounded_and_replayable(self):
        from src.methods.sera import SeraParams

        def merge(i, params, c, f=None):
            return MergeOutput({"w": torch.tensor(params.energy_threshold)})

        protocol = SelectionSpec("bayes", ({"alpha": 1.},), 100,
                                 max_trials=5, ei_tolerance=0)
        method = definition(SeraParams, merge, sera.select)
        result = sera.select(self.inputs, method, protocol, FakeValidation(peak=.65), self.context)
        observations = [e for e in result.history if e["event"] == "energy_observation"]
        self.assertEqual(len(observations), 5)
        validation = FakeValidation(peak=.65)
        replay = sera.select(self.inputs, method, protocol, validation, self.context,
                             resume_history=result.history, resume_output=result.output)
        self.assertFalse(validation.calls)
        self.assertEqual(replay.params, result.params)
        self.assertTrue(torch.equal(replay.output.weights["w"], result.output.weights["w"]))

    def test_sera_ei_finite_for_constant_observations(self):
        import numpy as np
        acquisition = sera.expected_improvement(np.array([.2, .5, .8]), np.ones(3), np.linspace(.1, .99, 20))
        self.assertTrue(np.isfinite(acquisition).all())
        self.assertTrue((acquisition >= 0).all())

    def test_validation_cannot_mutate_candidate_weights(self):
        class MutatingValidation(FakeValidation):
            def evaluate(self, weights, task_ids=None):
                result = super().evaluate(weights, task_ids)
                weights["w"].add_(10)
                return result

        with self.assertRaisesRegex(ValueError, "mutated"):
            search.select(self.inputs, definition(), SelectionSpec(), MutatingValidation(), self.context)


if __name__ == "__main__":
    unittest.main()
