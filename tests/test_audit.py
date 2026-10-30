"""Adversarial regression cases found in the post-migration audit."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.core.checkpoints import make_inputs, tensor_fingerprint
from src.core.types import (ExperimentSpec, MergeContext, MergeOutput, RuntimeSpec,
                            SelectionSpec, TaskSpec, EvaluationResult, TaskMetrics)
from src.experiments.registry import get_method
from src.experiments.storage import RunStorage


class RandomImages(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor([random.random(), np.random.random(), float(torch.rand(()))]), 0


class RecordingAdapter:
    identity = {"adapter": "record-random-inputs"}

    def __init__(self):
        self.images = []

    def load_weights(self, weights):
        pass

    def logits(self, images, task):
        self.images.append(images.clone())
        return torch.stack((torch.ones(len(images)), torch.zeros(len(images))), dim=1)


class AuditTests(unittest.TestCase):
    def test_fer2013_relative_dataset_root_is_applied_once(self):
        from src.datasets.adapter import _legacy
        with patch("src.datasets.adapter._hf_disk", return_value=(object(), ("a", "b"))) as factory:
            _legacy({"name": "FER2013", "split": "test"}, Path("datasets"), None, RuntimeSpec())
        specification, root, *_ = factory.call_args.args
        self.assertEqual(root / specification["path"], Path("datasets/fer2013_dataset/test"))

    def test_preprocessing_identity_is_stable_across_fresh_processes(self):
        import subprocess, sys
        root = Path(__file__).resolve().parents[1]
        code = (
            "import json; from src.models.adapter import preprocessing_identity; "
            "from open_clip.transform import image_transform; "
            "print(json.dumps(preprocessing_identity(image_transform(224, is_train=False)), sort_keys=True))"
        )
        outputs = [subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                                  text=True, capture_output=True, check=True).stdout for _ in range(2)]
        self.assertEqual(outputs[0], outputs[1])
        self.assertNotRegex(outputs[0], r"0x[0-9a-fA-F]+")

    def test_preprocessing_identity_distinguishes_settings_code_and_closures(self):
        from src.models.adapter import preprocessing_identity
        from torchvision.transforms import Compose, Lambda, Normalize
        first = preprocessing_identity(Compose([Normalize((.1,), (.2,))]))
        second = preprocessing_identity(Compose([Normalize((.1,), (.3,))]))
        self.assertNotEqual(first, second)
        one, two = (lambda image: image + 1), (lambda image: image + 2)
        self.assertNotEqual(preprocessing_identity(Lambda(one)), preprocessing_identity(Lambda(two)))
        def transform(value):
            return lambda image: image + value
        self.assertNotEqual(preprocessing_identity(Lambda(transform(1))), preprocessing_identity(Lambda(transform(2))))
        with self.assertRaisesRegex(TypeError, "Unsupported"):
            preprocessing_identity(object())

    def test_missing_acknowledged_journal_cannot_restart_budget(self):
        with tempfile.TemporaryDirectory() as root:
            storage = RunStorage(root, {"input": "same"})
            path = storage.path
            storage.observe({"sequence": 0, "event": "evaluation_start", "expected_images": 100})
            storage.close()
            (path / "trials.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "journal"):
                with RunStorage(root, {"input": "same"}, run_dir=path, resume=True):
                    pass

    def test_resume_checks_completion_after_acquiring_lock(self):
        import fcntl
        with tempfile.TemporaryDirectory() as root:
            storage = RunStorage(root, {"input": "same"})
            path = storage.path
            storage.close()
            original = fcntl.flock
            def concurrent_completion(descriptor, operation):
                if operation & fcntl.LOCK_EX:
                    (path / "result.json").write_text('{"status":"complete"}')
                return original(descriptor, operation)
            with patch("src.experiments.storage.fcntl.flock", side_effect=concurrent_completion):
                with self.assertRaises(FileExistsError):
                    with RunStorage(root, {"input": "same"}, run_dir=path, resume=True):
                        pass

    def test_saved_trace_arrays_are_checked_on_resume(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as root:
            storage = RunStorage(root, {"input": "same"})
            output = MergeOutput({"w": torch.tensor(1.)}, {"spectrum": torch.tensor([1., 2.])})
            storage.observe({"sequence": 0, "event": "evaluation_complete", "is_best": True}, output)
            event = storage.history[-1]
            save_file({"trace/spectrum": torch.tensor([50., 60.])}, str(storage.path / event["trace_file"]))
            with self.assertRaisesRegex(ValueError, "trace|Trace"):
                storage.resume_output()
            storage.close()

    def test_progress_test_metrics_are_checked_on_resume(self):
        with tempfile.TemporaryDirectory() as root:
            storage = RunStorage(root, {"input": "same"})
            path = storage.path
            storage.mark("test", test_metrics={"top1": .8})
            storage.close()
            progress = json.loads((path / "progress.json").read_text())
            progress["test_metrics"]["top1"] = .9
            (path / "progress.json").write_text(json.dumps(progress))
            with self.assertRaisesRegex(ValueError, "checksum"):
                with RunStorage(root, {"input": "same"}, run_dir=path, resume=True):
                    pass

    def test_random_baseline_does_not_depend_on_setup_or_cache_rng_consumption(self):
        inputs = make_inputs({"matrix": torch.zeros(8, 8)}, {"A": {"matrix": torch.eye(8)}})
        method = get_method("dummy")
        torch.manual_seed(7)
        one = method.merge(inputs, method.parameters({}), MergeContext(RuntimeSpec(seed=7)))
        torch.manual_seed(123)
        torch.rand(117)  # A model constructor or an uncached baseline consumed RNG.
        two = method.merge(inputs, method.parameters({}), MergeContext(RuntimeSpec(seed=7)))
        self.assertEqual(tensor_fingerprint(one.weights), tensor_fingerprint(two.weights))

    def test_evaluation_repeats_inputs_and_restores_all_global_rng_states(self):
        from src.datasets.adapter import EvaluationTask
        from src.evaluation.evaluator import Evaluator
        adapter = RecordingAdapter()
        task = EvaluationTask("A", "validation", DataLoader(RandomImages(), batch_size=4), ("a", "b"), {"sample_digest": "random-v1"})
        evaluator = Evaluator(adapter, (task,), RuntimeSpec(seed=9))
        random.seed(73); np.random.seed(73); torch.manual_seed(73)
        torch_state = torch.get_rng_state().clone()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        evaluator.evaluate({})
        first = torch.cat(adapter.images)
        adapter.images.clear()
        evaluator.evaluate({})
        self.assertTrue(torch.equal(first, torch.cat(adapter.images)))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))

    def test_normalizer_identity_includes_seed_and_workers(self):
        from src.datasets.adapter import EvaluationTask
        from src.evaluation.evaluator import Evaluator
        task = EvaluationTask("A", "validation", DataLoader(RandomImages()), ("a", "b"), {"sample_digest": "same"})
        one = Evaluator(RecordingAdapter(), (task,), RuntimeSpec(seed=0, num_workers=0))
        two = Evaluator(RecordingAdapter(), (task,), RuntimeSpec(seed=1, num_workers=0))
        three = Evaluator(RecordingAdapter(), (task,), RuntimeSpec(seed=0, num_workers=1))
        self.assertNotEqual(one.identity, two.identity)
        self.assertNotEqual(one.identity, three.identity)

    def test_individual_saved_config_describes_the_actual_child_execution(self):
        from src.evaluation.single_task import run_individuals
        tasks = tuple(TaskSpec(t, {}, {}, "head") for t in ("A", "B"))
        spec = ExperimentSpec("synthetic", tasks, {"A": "a", "B": "b"}, "base", "data", "tsvm", {}, SelectionSpec(), RuntimeSpec(),
                              resolved_config={"benchmark": {"name": "two", "task_ids": ["A", "B"], "tasks": {}},
                                               "method": {"id": "tsvm", "parameters": {}},
                                               "protocol": {"name": "common_fixed", "selection": {"mode": "fixed"}}})
        calls = []
        run_individuals(spec, run=lambda child, **kw: calls.append(child))
        for child in calls:
            self.assertEqual(child.resolved_config["method"]["id"], child.method)
            self.assertEqual(child.resolved_config["benchmark"]["task_ids"], [child.tasks[0].id])
        self.assertEqual(spec.resolved_config["method"]["id"], "tsvm")

    def test_comparison_does_not_merge_different_source_revisions(self):
        from src.experiments.analysis import check_comparability
        record = {"config": {"model": "synthetic", "benchmark": "same"}, "input_identity": {"same": True},
                  "provenance": {"source_sha256": "revision-one"}}
        other = {**record, "provenance": {"source_sha256": "revision-two"}}
        with self.assertRaises(ValueError):
            check_comparability([record, other])

    def test_validation_and_test_cannot_use_identical_observed_datasets(self):
        from src.experiments.runner import run_experiment
        inputs = make_inputs({"w": torch.tensor(0.)}, {"A": {"w": torch.tensor(1.)}})
        from src.core.types import PreparedExperiment
        with tempfile.TemporaryDirectory() as root:
            spec = ExperimentSpec("synthetic", (), {}, "", "", "task_arithmetic", {}, SelectionSpec(), RuntimeSpec(output_dir=root))
            prepared = PreparedExperiment(spec, inputs, {"merge_inputs": inputs.identity})
            class Bound:
                task_ids = ("A",)
                task_sizes = {"A": 1}
                def __init__(self, split):
                    self.split = split
                    self.identity = {"split": split, "tasks": {"A": {"sample_digest": "identical-inputs"}}}
                def evaluate(self, weights, task_ids=None):
                    return EvaluationResult(self.split, {"A": TaskMetrics(1, 1, 1., 1.)}, 1., 1., 1)
            with self.assertRaisesRegex(ValueError, "identical|overlap"):
                run_experiment(spec, prepared=prepared, evaluator_factory=lambda prepared, split: Bound(split))

    def test_method_additions_keep_comparable_shared_infrastructure(self):
        from src.experiments.analysis import check_comparability
        record = {"config": {"model": "synthetic", "benchmark": "same"}, "input_identity": {},
                  "provenance": {"source_sha256": "method-checkout-one", "common_sha256": "same-shared-code"}}
        other = {**record, "provenance": {"source_sha256": "method-checkout-two", "common_sha256": "same-shared-code"}}
        self.assertTrue(check_comparability([record, other])["comparable"])

    def test_partial_file_overlap_is_detected_before_selection(self):
        from PIL import Image
        from src.datasets.adapter import inspect_task
        from src.experiments.runner import check_split_separation
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for split in ("validation", "test"):
                (root / split / "a").mkdir(parents=True)
            Image.new("RGB", (2, 2), "red").save(root / "validation/a/one.png")
            Image.new("RGB", (2, 2), "blue").save(root / "validation/a/shared.png")
            Image.new("RGB", (2, 2), "green").save(root / "test/a/two.png")
            (root / "test/a/shared.png").symlink_to(root / "validation/a/shared.png")
            spec = TaskSpec("A", {"adapter": "imagefolder", "path": "validation"},
                            {"adapter": "imagefolder", "path": "test"}, "unused-head")
            first = inspect_task(spec, "validation", str(root), None, RuntimeSpec())
            second = inspect_task(spec, "test", str(root), None, RuntimeSpec())
            self.assertNotEqual(first.identity["sample_digest"], second.identity["sample_digest"])
            one = SimpleNamespace(split="validation", task_ids=("A",), tasks={"A": first}, identity={"tasks": {"A": first.identity}})
            two = SimpleNamespace(split="test", task_ids=("A",), tasks={"A": second}, identity={"tasks": {"A": second.identity}})
            with self.assertRaisesRegex(ValueError, "overlap"):
                check_split_separation(one, two)

    def test_sera_proposals_use_the_declared_selection_objective(self):
        from src.selection import sera
        from src.core.types import MethodDefinition
        from src.methods.sera import SeraParams
        inputs = make_inputs({"w": torch.tensor(0.)}, {"A": {"w": torch.tensor(1.)}, "B": {"w": torch.tensor(2.)}})
        class Validation:
            split = "validation"
            task_ids = ("A", "B")
            task_sizes = {"A": 10, "B": 10}
            def evaluate(self, weights, task_ids=None):
                return EvaluationResult("validation", {"A": TaskMetrics(8, 10, .8, .8), "B": TaskMetrics(2, 10, .2, .2)}, .5, .5, 20)
        method = MethodDefinition("synthetic", lambda *args: MergeOutput({"w": torch.tensor(1.)}),
                                  SeraParams, sera.select, "test", "synthetic")
        protocol = SelectionSpec(mode="bayes", max_validation_images=20, max_trials=1,
                                 initial_energies=(.5,), objective="first_task_top1")
        selected = sera.select(inputs, method, protocol, Validation(), MergeContext(RuntimeSpec()))
        observations = [event for event in selected.history if event["event"] == "energy_observation"]
        self.assertEqual(observations[0]["score"], .8)

    def test_published_result_metrics_cannot_be_silently_modified(self):
        from src.experiments.analysis import load_results
        with tempfile.TemporaryDirectory() as root:
            storage = RunStorage(root, {"input": "same"})
            path = storage.finish({"schema_version": 1, "status": "complete", "test": {"avg_top1": .8}}, MergeOutput({"w": torch.tensor(1.)}))
            storage.close()
            record = json.loads(path.read_text())
            record["test"]["avg_top1"] = .99
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_results([path])

    def test_cli_works_from_a_numpy_initialized_parent_process(self):
        import subprocess, sys, os
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, MKL_THREADING_LAYER="INTEL")
        result = subprocess.run([sys.executable, "-B", str(root / "run.py"), "--cfg", "job"],
                                cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tensor_observations_are_supported_and_cannot_mutate_the_cache(self):
        from dataclasses import dataclass
        from src.core.numerics import cached_merge
        @dataclass(frozen=True)
        class Params:
            alpha: float = 1.
        inputs = make_inputs({"w": torch.tensor(0.)}, {"A": {"w": torch.tensor(1.)}})
        context = MergeContext(RuntimeSpec(trace="full"))
        def build():
            return {"w": torch.tensor(1.)}, {"spectrum": torch.tensor([1., 2.])}
        first = cached_merge(inputs, Params(), context, "tensor-trace", build)
        first.trace["spectrum"].zero_()
        second = cached_merge(inputs, Params(alpha=.5), context, "tensor-trace", build)
        self.assertTrue(torch.equal(second.trace["spectrum"], torch.tensor([1., 2.])))
        self.assertEqual(second.weights["w"].item(), .5)

    def test_stochastic_input_is_replayed_with_multiple_loader_workers(self):
        from src.datasets.adapter import EvaluationTask
        from src.evaluation.evaluator import Evaluator
        adapter = RecordingAdapter()
        task = EvaluationTask("A", "validation", DataLoader(RandomImages(), batch_size=4, num_workers=2), ("a", "b"), {"sample_digest": "random-v1"})
        evaluator = Evaluator(adapter, (task,), RuntimeSpec(seed=9, num_workers=2))
        evaluator.evaluate({})
        first = torch.cat(adapter.images)
        adapter.images.clear()
        evaluator.evaluate({})
        self.assertTrue(torch.equal(first, torch.cat(adapter.images)))

    def test_conflicting_plot_destination_does_not_leave_a_csv(self):
        from src.experiments.analysis import load_results
        import compare
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "comparison.csv"
            figures = Path(root) / "figures"; figures.mkdir()
            record = {"config": {"model": "x", "benchmark": "same"}, "input_identity": {}, "method": {"id": "x"},
                      "test": {"avg_top1": 1., "avg_normalized_top1": 1.}, "selected_params": {}, "weights_fingerprint": "x"}
            with patch("sys.argv", ["compare.py", "unused", "--output", str(output), "--plots", str(figures)]), patch("compare.load_results", return_value=[record]):
                with self.assertRaises(FileExistsError):
                    compare.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
