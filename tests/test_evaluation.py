"""Evaluation tests use synthetic inputs and never download datasets/models."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.types import MergeInputs, PreparedExperiment, RuntimeSpec, TaskSpec
from src.datasets.adapter import EvaluationTask, dataset_digest, inspect_task
from src.evaluation.evaluator import Evaluator
from src.evaluation.normalization import get_baselines
from src.models.adapter import OpenClipAdapter, load_head


class MemorySource:
    def __init__(self, weights, identity):
        self.weights, self.identity = weights, identity

    def keys(self):
        return tuple(self.weights)

    def tensor(self, name):
        return self.weights[name]


class LogitAdapter:
    identity = {"adapter": "synthetic-linear-v1", "preprocess": "identity"}
    preprocess = None

    def __init__(self):
        self.loads = 0

    def load_weights(self, weights):
        self.weights = weights
        self.loads += 1

    def logits(self, images, task_id):
        return images * self.weights["scale"]

    def head_identity(self, task_id, class_names, declared_names=()):
        return {"head": "synthetic-head", "class_names": class_names}


def task(task_id, images, labels, split="validation"):
    dataset = TensorDataset(torch.tensor(images, dtype=torch.float32), torch.tensor(labels))
    return EvaluationTask(task_id, split, DataLoader(dataset, batch_size=2), ("zero", "one"), {
        "task": task_id, "split": split, "sample_digest": dataset_digest(dataset, "."),
        "head": {"hash": "provided-head"},
    })


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tasks = (
            task("A", [[3, 1], [3, 1], [1, 3]], [0, 1, 1]),
            task("B", [[1, 3]], [1]),
        )
        self.runtime = RuntimeSpec()

    def test_macro_accuracy_and_mean_of_ratios(self):
        evaluator = Evaluator(LogitAdapter(), self.tasks, self.runtime, {"A": 1 / 3, "B": 1.0})
        weights = {"scale": torch.tensor(1.0)}
        result = evaluator.evaluate(weights)
        self.assertEqual(result.per_task["A"].correct, 2)
        self.assertEqual(result.evaluated_images, 4)
        self.assertAlmostEqual(result.avg_top1, (2 / 3 + 1) / 2)
        self.assertAlmostEqual(result.avg_normalized_top1, 1.5)
        self.assertEqual(weights["scale"].item(), 1)

    def test_bound_split_and_task_subset(self):
        evaluator = Evaluator(LogitAdapter(), self.tasks, self.runtime)
        result = evaluator.evaluate({"scale": torch.tensor(1.)}, ("B",))
        self.assertEqual(result.split, "validation")
        self.assertEqual(result.evaluated_images, 1)
        with self.assertRaises(AttributeError):
            evaluator.split = "test"
        with self.assertRaises(ValueError):
            evaluator.evaluate({"scale": torch.tensor(1.)}, ("unknown",))
        with self.assertRaises(ValueError):
            Evaluator(LogitAdapter(), (self.tasks[0], task("C", [[1, 2]], [1], "test")), self.runtime)

    def test_invalid_normalizers_and_outputs_fail(self):
        for baselines in ({"A": 0, "B": 1}, {"A": .5}):
            with self.assertRaises(ValueError):
                Evaluator(LogitAdapter(), self.tasks, self.runtime, baselines)
        evaluator = Evaluator(LogitAdapter(), self.tasks, self.runtime)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            evaluator.evaluate({"scale": torch.tensor(float("nan"))})

    def test_normalizer_cache_binds_checkpoint_data_head_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = LogitAdapter()
            evaluator = Evaluator(adapter, self.tasks, self.runtime)
            positive = MemorySource({"scale": torch.tensor(1.)}, "checkpoint-positive")
            inputs = MergeInputs(positive, {"A": positive, "B": positive}, ("A", "B"), ("scale",), "input")
            prepared = PreparedExperiment(types.SimpleNamespace(runtime=self.runtime), inputs, {})
            first = get_baselines(prepared, "validation", evaluator, directory)
            loads = adapter.loads
            self.assertEqual(get_baselines(prepared, "validation", evaluator, directory), first)
            self.assertEqual(adapter.loads, loads)
            self.assertEqual(len(list(Path(directory).rglob("*.json"))), 2)
            self.tasks[0].identity["head"]["hash"] = "different-head"
            get_baselines(prepared, "validation", evaluator, directory)
            self.assertEqual(adapter.loads, loads + 1)
            self.assertEqual(len(list(Path(directory).rglob("*.json"))), 3)
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                get_baselines(prepared, "test", evaluator, directory)

    def test_corrupt_normalizer_cache_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = Evaluator(LogitAdapter(), self.tasks[:1], self.runtime)
            source = MemorySource({"scale": torch.tensor(1.)}, "checkpoint")
            inputs = MergeInputs(source, {"A": source}, ("A",), ("scale",), "input")
            prepared = PreparedExperiment(types.SimpleNamespace(runtime=self.runtime), inputs, {})
            get_baselines(prepared, "validation", evaluator, directory)
            path = next(Path(directory).rglob("*.json"))
            content = json.loads(path.read_text())
            content["identity"]["finetuned"] = "wrong-checkpoint"
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                get_baselines(prepared, "validation", evaluator, directory)

    def test_tensor_dataset_class_order_and_split_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.save({"images": torch.ones(2, 2), "labels": torch.tensor([0, 1]), "class_names": ["cat", "dog"]}, Path(directory) / "images.pt")
            description = {"adapter": "tensor_file", "path": "images.pt"}
            spec = TaskSpec("A", description, description, "head.pt", ("cat", "dog"))
            actual = inspect_task(spec, "validation", directory, None, self.runtime)
            self.assertEqual(actual.size, 2)
            mismatch = TaskSpec("A", description, description, "head.pt", ("dog", "cat"))
            with self.assertRaisesRegex(ValueError, "class order"):
                inspect_task(mismatch, "validation", directory, None, self.runtime)
            bad_split = TaskSpec("A", {**description, "split": "test"}, description, "head.pt")
            with self.assertRaisesRegex(ValueError, "declares test"):
                inspect_task(bad_split, "validation", directory, None, self.runtime)

    def test_normalizer_accuracy_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = Evaluator(LogitAdapter(), self.tasks[:1], self.runtime)
            source = MemorySource({"scale": torch.tensor(1.)}, "checkpoint")
            inputs = MergeInputs(source, {"A": source}, ("A",), ("scale",), "input")
            prepared = PreparedExperiment(types.SimpleNamespace(runtime=self.runtime), inputs, {})
            get_baselines(prepared, "validation", evaluator, directory)
            path = next(Path(directory).rglob("*.json"))
            record = json.loads(path.read_text())
            record["accuracy"] = .1
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "checksum"):
                get_baselines(prepared, "validation", evaluator, directory)

    def test_actual_normalizers_are_part_of_evaluation_identity(self):
        first = Evaluator(LogitAdapter(), self.tasks, self.runtime, {"A": .5, "B": 1.})
        second = Evaluator(LogitAdapter(), self.tasks, self.runtime, {"A": .8, "B": 1.})
        self.assertNotEqual(first.identity, second.identity)

    def test_file_backed_dataset_fingerprint_changes_with_bytes_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(b"first image")
            dataset = types.SimpleNamespace(_image_files=[path], _labels=[0])
            first = dataset_digest(dataset, directory)
            path.write_bytes(b"changed image")
            second = dataset_digest(dataset, directory)
            self.assertNotEqual(first, second)
            dataset._labels = [1]
            self.assertNotEqual(second, dataset_digest(dataset, directory))

    def test_validation_partition_matches_legacy_seed_zero(self):
        from src.datasets.adapter import _validation_subset
        dataset = TensorDataset(torch.arange(60000))
        subset = _validation_subset(dataset, {})
        train, reference = torch.utils.data.random_split(dataset, [55000, 5000], generator=torch.Generator().manual_seed(0))
        self.assertEqual(subset.indices, reference.indices)
        self.assertEqual(train.indices[0], 36044)

    def test_offline_legacy_constructor_passes_download_false(self):
        from src.datasets.adapter import _legacy
        from types import SimpleNamespace
        calls = []
        def loader(name, preprocess, **kwargs):
            calls.append(kwargs["download"])
            return SimpleNamespace(test_dataset=TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long)), classnames=("a",))
        with patch("src.datasets.registry.get_dataset", side_effect=loader):
            _legacy({"name": "MNIST", "split": "validation"}, Path("unused"), None, self.runtime)
        self.assertEqual(calls, [False])

    def test_unknown_legacy_profile_cannot_silently_reuse_local_splits(self):
        from src.datasets.adapter import _legacy
        with self.assertRaisesRegex(ValueError, "Unknown legacy dataset profile"):
            _legacy({"name": "MNIST", "split": "validation", "profile": "paper_exact"},
                    Path("unused"), None, self.runtime)

    def test_head_metadata_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            torch.save({"weight": torch.eye(2), "class_names": ["cat", "dog"], "normalize": False}, path)
            head = load_head(path)
            self.assertFalse(head.normalize)
            self.assertEqual(head.class_names, ("cat", "dog"))
            self.assertTrue(torch.equal(head.bias, torch.zeros(2)))

    def test_model_construction_is_offline_and_head_semantics_are_explicit(self):
        class FakeClip(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))
                self.transformer = torch.nn.Linear(2, 2)
                self.visual = types.SimpleNamespace(output_dim=2)

            def encode_image(self, images):
                return images @ self.weight

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            torch.save({"weight": torch.eye(2), "normalize": False}, path)
            spec = types.SimpleNamespace(
                model="ViT-B-32", pretrained_tag="openai", runtime=self.runtime,
                tasks=(TaskSpec("A", {}, {}, str(path)),),
            )
            with patch("open_clip.create_model_and_transforms", return_value=(FakeClip(), None, None)) as factory:
                adapter = OpenClipAdapter(spec, ("model.weight",))
            self.assertIsNone(factory.call_args.kwargs["pretrained"])
            self.assertFalse(hasattr(adapter.encoder.model, "transformer"))
            adapter.load_weights({"model.weight": torch.eye(2)})
            self.assertTrue(torch.equal(adapter.logits(torch.tensor([[3., 1.]]), "A"), torch.tensor([[3., 1.]])))
            identity = adapter.head_identity("A", ("cat", "dog"))
            self.assertEqual(identity["class_order_status"], "unverified_author_head")
            with self.assertRaisesRegex(ValueError, "head has 2 classes"):
                adapter.head_identity("A", ("one",))
            with self.assertRaisesRegex(ValueError, "between evaluation splits"):
                adapter.head_identity("A", ("dog", "cat"))
            self.assertEqual(adapter.head_identity("A", ("cat", "dog"))["class_order_status"],
                             "unverified_author_head")

    def test_build_evaluator_populates_only_its_split_baselines(self):
        from src.evaluation.evaluator import build_evaluator
        with tempfile.TemporaryDirectory() as directory:
            for split in ("validation", "test"):
                torch.save({"images": torch.tensor([[3., 1.], [1., 3.]]),
                            "labels": torch.tensor([0, 1]), "class_names": ["zero", "one"]},
                           Path(directory) / f"{split}.pt")
            task_spec = TaskSpec("A", {"adapter": "tensor_file", "path": "validation.pt"},
                                 {"adapter": "tensor_file", "path": "test.pt"}, "unused-head")
            runtime = RuntimeSpec(cache_dir=directory)
            spec = types.SimpleNamespace(runtime=runtime, tasks=(task_spec,), data_root=directory)
            source = MemorySource({"scale": torch.tensor(1.)}, "ft")
            inputs = MergeInputs(source, {"A": source}, ("A",), ("scale",), "inputs")
            prepared = PreparedExperiment(spec, inputs, {})
            adapter = LogitAdapter()
            evaluator = build_evaluator(prepared, "validation", adapter)
            self.assertEqual(evaluator.evaluate(source.weights).avg_normalized_top1, 1.)
            self.assertEqual(len(list(Path(directory).rglob("*.json"))), 1)
            test_evaluator = build_evaluator(prepared, "test", adapter)
            self.assertEqual(test_evaluator.evaluate(source.weights).avg_normalized_top1, 1.)
            self.assertEqual(len(list(Path(directory).rglob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
