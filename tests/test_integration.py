"""End-to-end selection/test/storage without real datasets or pretrained downloads."""
from pathlib import Path
import tempfile
import unittest

import torch

from src.core.checkpoints import make_inputs, tensor_fingerprint
from src.core.types import ExperimentSpec, PreparedExperiment, RuntimeSpec, SelectionSpec
from src.experiments.analysis import load_results, make_comparison_table
from src.experiments.runner import run_experiment
from test_selection import FakeValidation


class IntegrationTests(unittest.TestCase):
    def fixture(self, root):
        inputs = make_inputs({"w": torch.tensor(0.)}, {
            "A": {"w": torch.tensor(1.)}, "B": {"w": torch.tensor(1.)}})
        spec = ExperimentSpec("synthetic", (), {}, "", "", "task_arithmetic", {},
                              SelectionSpec("grid", ({"alpha": .25}, {"alpha": .5}, {"alpha": 1.}), 60),
                              RuntimeSpec(output_dir=root), benchmark="synthetic")
        return spec, PreparedExperiment(spec, inputs, {"merge_inputs": inputs.identity})

    def test_test_receives_selected_weights_once_and_complete_bundle_is_readable(self):
        with tempfile.TemporaryDirectory() as root:
            inputs = make_inputs({"w": torch.tensor(0.)}, {
                "A": {"w": torch.tensor(1.)}, "B": {"w": torch.tensor(1.)}})
            spec = ExperimentSpec("synthetic", (), {}, "", "", "task_arithmetic", {},
                                  SelectionSpec("grid", ({"alpha": .25}, {"alpha": .5}, {"alpha": 1.}), 60),
                                  RuntimeSpec(output_dir=root), benchmark="synthetic")
            prepared = PreparedExperiment(spec, inputs, {"merge_inputs": inputs.identity})
            calls = []

            class BoundEvaluator(FakeValidation):
                def __init__(self, split):
                    super().__init__()
                    self.split = split
                    self.identity = {"split": split, "fixture": "v1"}

                def evaluate(self, weights, task_ids=None):
                    from dataclasses import replace
                    result = super().evaluate(weights, task_ids)
                    calls.append((self.split, tensor_fingerprint(weights)))
                    return replace(result, split=self.split)

            def factory(prepared, split, **kwargs):
                return BoundEvaluator(split)

            result = run_experiment(spec, prepared=prepared, evaluator_factory=factory)
            self.assertEqual(result["selected_params"]["alpha"], .5)
            test_calls = [digest for split, digest in calls if split == "test"]
            self.assertEqual(test_calls, [result["weights_fingerprint"]])
            self.assertIn(result["weights_fingerprint"], [digest for split, digest in calls if split == "validation"])
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "selected.safetensors").is_file())
            records = load_results([run_dir])
            self.assertEqual(make_comparison_table(records)[0]["method"], "task_arithmetic")
            with self.assertRaises(FileExistsError):
                run_experiment(spec, prepared=prepared, evaluator_factory=factory, run_dir=run_dir, resume=True)

    def test_interrupted_test_resumes_frozen_model_without_validation_replay(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as root:
            spec, prepared = self.fixture(root)
            path = Path(root) / "resumable"
            evaluations = []
            fail = [True]

            class Bound(FakeValidation):
                def __init__(self, split):
                    super().__init__()
                    self.split = split
                    self.identity = {"split": split, "data": "same"}

                def evaluate(self, weights, task_ids=None):
                    evaluations.append((self.split, tensor_fingerprint(weights)))
                    if self.split == "test" and fail[0]:
                        fail[0] = False
                        raise RuntimeError("interrupted test")
                    return replace(super().evaluate(weights, task_ids), split=self.split)

            factory = lambda prepared, split, **kwargs: Bound(split)
            with self.assertRaisesRegex(RuntimeError, "interrupted test"):
                run_experiment(spec, prepared=prepared, evaluator_factory=factory, run_dir=path)
            self.assertEqual(sum(split == "validation" for split, _ in evaluations), 3)
            result = run_experiment(spec, prepared=prepared, evaluator_factory=factory, run_dir=path, resume=True)
            self.assertEqual(sum(split == "validation" for split, _ in evaluations), 3)
            self.assertEqual(result["selected_params"]["alpha"], .5)
            self.assertEqual(evaluations[-1], ("test", result["weights_fingerprint"]))

    def test_test_split_change_rejected_before_resumed_selection(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as root:
            spec, prepared = self.fixture(root)
            path = Path(root) / "changed-data"
            revision = ["one"]
            calls = []

            class Bound(FakeValidation):
                def __init__(self, split):
                    super().__init__()
                    self.split = split
                    self.identity = {"split": split, "data": revision[0] if split == "test" else "same-val"}

                def evaluate(self, weights, task_ids=None):
                    calls.append(self.split)
                    if self.split == "test":
                        raise RuntimeError("stop before publication")
                    return replace(super().evaluate(weights, task_ids), split=self.split)

            factory = lambda prepared, split, **kwargs: Bound(split)
            with self.assertRaisesRegex(RuntimeError, "publication"):
                run_experiment(spec, prepared=prepared, evaluator_factory=factory, run_dir=path)
            before = list(calls)
            revision[0] = "different-test-images"
            with self.assertRaisesRegex(ValueError, "identity"):
                run_experiment(spec, prepared=prepared, evaluator_factory=factory, run_dir=path, resume=True)
            self.assertEqual(calls, before)


if __name__ == "__main__":
    unittest.main()
