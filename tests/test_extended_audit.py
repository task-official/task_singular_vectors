"""Independent failure cases for dataset identity and interrupted evaluation."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.core.checkpoints import make_inputs
from src.core.types import (EvaluationResult, ExperimentSpec, PreparedExperiment,
                            RuntimeSpec, SelectionSpec, TaskMetrics)


class ExtendedAuditTests(unittest.TestCase):
    def test_boolean_method_and_runtime_settings_are_not_truthy_strings(self):
        from src.experiments.registry import get_method
        for method, values in (("subspace_tsv", {"weighted": "false"}),
                               ("individual", {"reconstructed": "false"})):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "boolean"):
                get_method(method).parameters(values)
        with self.assertRaisesRegex(ValueError, "boolean"):
            RuntimeSpec(save_weights="false")

    def test_mnist_validation_accepts_an_explicit_different_seed(self):
        from src.datasets.registry import split_train_into_train_val
        from src.datasets.splits import validation_partition
        dataset = torch.utils.data.TensorDataset(torch.arange(60000))
        bundle = SimpleNamespace(train_dataset=dataset, classnames=["a"])
        actual = split_train_into_train_val(bundle, "MNISTVal", 32, 0, .1, 5000, seed=1)
        expected = validation_partition(dataset, .1, 5000, seed=1)[1]
        self.assertEqual(actual.test_dataset.indices, expected.indices)

    def _hf_images(self, root):
        from datasets import Dataset, load_from_disk, disable_progress_bar
        from PIL import Image
        disable_progress_bar()
        paths = [root / f"image-{i}.png" for i in range(4)]
        for path, color in zip(paths, ("red", "blue", "green", "yellow")):
            Image.new("RGB", (2, 2), color).save(path)
        Dataset.from_dict({"image": [{"path": str(p), "bytes": None} for p in paths],
                           "label": [0, 1, 0, 1]}).save_to_disk(str(root / "hf"))
        return load_from_disk(str(root / "hf")), paths

    def test_hf_identity_tracks_the_logical_row_selection_and_order(self):
        from src.datasets.adapter import _HFDiskImages, dataset_digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, _ = self._hf_images(root)
            first = _HFDiskImages(data.select([0, 1]), None)
            reverse = _HFDiskImages(data.select([1, 0]), None)
            other = _HFDiskImages(data.select([2, 3]), None)
            self.assertNotEqual(dataset_digest(first, root), dataset_digest(reverse, root))
            self.assertNotEqual(dataset_digest(first, root), dataset_digest(other, root))

    def test_hf_identity_tracks_external_image_bytes(self):
        from src.datasets.adapter import _HFDiskImages, dataset_digest
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, paths = self._hf_images(root)
            wrapped = _HFDiskImages(data, None)
            first = dataset_digest(wrapped, root)
            Image.new("RGB", (2, 2), "black").save(paths[0])
            self.assertNotEqual(first, dataset_digest(wrapped, root))

    def test_hf_partial_overlap_has_membership_evidence(self):
        from src.datasets.adapter import _HFDiskImages, sample_membership
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, paths = self._hf_images(root)
            first = sample_membership(_HFDiskImages(data.select([0, 1]), None), root, {})
            second = sample_membership(_HFDiskImages(data.select([1, 2]), None), root, {})
            self.assertIsNotNone(first)
            self.assertEqual(set(first).intersection(second), {"file:" + str(paths[1].resolve())})

    def test_hf_mixed_sources_keep_membership_aligned_through_subsets(self):
        from datasets import Dataset, Features, Image as HFImage, Value
        from PIL import Image
        from src.datasets.adapter import _HFDiskImages, sample_membership, dataset_digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, paths = self._hf_images(root)
            features = Features({"image": HFImage(), "label": Value("int64")})
            data = Dataset.from_dict({"image": [{"path": str(paths[0]), "bytes": None},
                                                {"path": None, "bytes": paths[1].read_bytes()},
                                                {"path": str(paths[2]), "bytes": None}],
                                      "label": [0, 1, 0]}, features=features)
            wrapped = _HFDiskImages(data, None)
            subset = torch.utils.data.Subset(wrapped, [1, 2])
            self.assertEqual(sample_membership(subset, root, {}), (None, "file:" + str(paths[2].resolve())))
            before = dataset_digest(wrapped, root)
            Image.new("RGB", (2, 2), "black").save(paths[2])
            self.assertNotEqual(before, dataset_digest(wrapped, root))

    def test_evaluation_metrics_are_checked_before_they_can_be_used(self):
        from src.core.types import validate_evaluation_result
        good = EvaluationResult("test", {"A": TaskMetrics(1, 2, .5, .5)}, .5, .5, 2)
        invalid = (
            replace(good, split="validation"),
            replace(good, per_task={}),
            replace(good, evaluated_images=1),
            replace(good, per_task={"A": TaskMetrics(3, 2, 1.5, 1.5)}),
            replace(good, per_task={"A": TaskMetrics(1, 2, .8, .5)}),
            replace(good, avg_top1=.8),
            replace(good, avg_normalized_top1=float("nan")),
            replace(good, avg_normalized_top1=None),
        )
        validate_evaluation_result(good, "test", {"A": 2})
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                validate_evaluation_result(record, "test", {"A": 2})

    def test_normalization_identity_distinguishes_gpu_and_cudnn_settings(self):
        from src.datasets.adapter import EvaluationTask
        from src.evaluation.evaluator import Evaluator
        data = torch.utils.data.TensorDataset(torch.eye(2), torch.tensor([0, 1]))
        task = EvaluationTask("A", "validation", torch.utils.data.DataLoader(data), ("a", "b"), {"data": "same"})
        adapter = SimpleNamespace(identity={"adapter": "same"})
        def identity(name, deterministic):
            gpu = SimpleNamespace(name=name, major=8, minor=6, total_memory=1234)
            with patch("torch.cuda.get_device_properties", return_value=gpu), \
                 torch.backends.cudnn.flags(deterministic=deterministic):
                return Evaluator(adapter, (task,), RuntimeSpec(device="cuda:0")).identity
        one = identity("GPU-one", False)
        self.assertNotEqual(one, identity("GPU-two", False))
        self.assertNotEqual(one, identity("GPU-one", True))

    def test_a_failed_test_evaluator_cannot_leave_reusable_metrics(self):
        from src.experiments.runner import run_experiment
        with tempfile.TemporaryDirectory() as directory:
            inputs = make_inputs({"w": torch.tensor(0.)}, {"A": {"w": torch.tensor(1.)}})
            spec = ExperimentSpec("synthetic", (), {}, "", "", "task_arithmetic", {},
                                  SelectionSpec(), RuntimeSpec(output_dir=directory))
            prepared = PreparedExperiment(spec, inputs, {"inputs": inputs.identity})
            path = Path(directory) / "run"
            broken = [True]
            test_calls = []

            class Evaluator:
                task_ids = ("A",)
                task_sizes = {"A": 2}

                def __init__(self, split):
                    self.split = split
                    self.identity = {"split": split, "version": "same"}

                def evaluate(self, weights, task_ids=None):
                    if self.split == "test":
                        test_calls.append(1)
                        if broken[0]:
                            weights["w"].add_(100)
                    correct = 2 if weights["w"].item() > 50 else 1
                    return EvaluationResult(self.split, {"A": TaskMetrics(correct, 2, correct/2, correct/2)},
                                            correct/2, correct/2, 2)

            factory = lambda prepared, split, **kw: Evaluator(split)
            with self.assertRaisesRegex(ValueError, "mutated"):
                run_experiment(spec, prepared=prepared, evaluator_factory=factory, run_dir=path)
            broken[0] = False
            result = run_experiment(spec, prepared=prepared, evaluator_factory=factory,
                                    run_dir=path, resume=True)
            self.assertEqual(len(test_calls), 2, "Invalid test metrics were reused on resume")
            self.assertEqual(result["test"]["avg_top1"], .5)

    def test_ambiguous_head_normalization_is_rejected(self):
        from src.models.adapter import load_head
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            torch.save({"weight": torch.eye(2), "normalize": "false"}, path)
            with self.assertRaisesRegex(ValueError, "normalize"):
                load_head(path)
            path = Path(directory) / "head.safetensors"
            save_file({"weight": torch.eye(2)}, str(path), metadata={"normalize": "flase"})
            with self.assertRaisesRegex(ValueError, "normalize"):
                load_head(path)

    def test_trace_tensor_paths_cannot_silently_overwrite_each_other(self):
        from src.experiments.storage import split_trace
        trace = {"layer/a": torch.tensor([1., 2.]), "layer": {"a": torch.tensor([3., 4.])}}
        encoded, tensors = split_trace(trace)
        one, two = encoded["layer/a"]["tensor"], encoded["layer"]["a"]["tensor"]
        self.assertNotEqual(one, two)
        self.assertTrue(torch.equal(tensors[one], trace["layer/a"]))
        self.assertTrue(torch.equal(tensors[two], trace["layer"]["a"]))

    def test_plots_distinguish_method_implementation_versions(self):
        from src.experiments.analysis import make_comparison_table
        from src.experiments.plotting import plot_comparison
        record = {"config": {"model": "synthetic", "benchmark": "same"}, "input_identity": {},
                  "selected_params": {"alpha": 1.}, "weights_fingerprint": "x",
                  "test": {"avg_top1": .8, "avg_normalized_top1": .9, "per_task": {}}}
        records = [{**record, "method": {"id": "same_method", "implementation": version}}
                   for version in ("version_one", "version_two")]
        labels = []
        def save(figure, *args, **kwargs):
            labels.append([label.get_text() for label in figure.axes[0].get_xticklabels()])
        with tempfile.TemporaryDirectory() as directory, patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=save):
            plot_comparison(records, make_comparison_table(records), Path(directory) / "figures")
        self.assertEqual(len(set(labels[0])), 2)
        self.assertIn("version_one", labels[0][0])
        self.assertIn("version_two", labels[0][1])

    def test_diagnostic_export_uses_the_actual_relocated_run_path(self):
        from src.core.types import MergeOutput
        from src.experiments.storage import RunStorage
        from scripts.collect_five_method_merge_diagnostics import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with RunStorage(root, {"input": "same"}) as storage:
                old = storage.path
                storage.finish({"schema_version": 1, "status": "complete", "method": {"id": "test"},
                                "run_dir": str(old), "test": {}},
                               MergeOutput({"w": torch.tensor(1.)}, {"basis": torch.eye(2)}))
            new = root / "relocated"
            old.rename(new)
            output = root / "export.json"
            with patch("sys.argv", ["collect", "--runs", str(new), "--output", str(output)]):
                main()
            record = json.loads(output.read_text())["observed_runs"][0]
            self.assertEqual(Path(record["trace_file"]), new / "trace.safetensors")
            self.assertTrue(Path(record["trace_file"]).is_file())
            from safetensors.torch import save_file
            save_file({"trace/basis": torch.ones(2, 2)}, str(new / "trace.safetensors"))
            output = root / "corrupt-export.json"
            with patch("sys.argv", ["collect", "--runs", str(new), "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
