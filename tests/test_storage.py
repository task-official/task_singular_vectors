import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from src.core.types import MergeOutput
from src.experiments.analysis import check_comparability, export_csv, make_comparison_table
from src.experiments.storage import RunStorage, identity_hash


class StorageTests(unittest.TestCase):
    def test_exclusive_creation_resume_identity_and_saved_weights(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "one"
            output = MergeOutput({"w": torch.tensor([1., 2.])}, {"basis": torch.eye(2)})
            storage = RunStorage(root, {"input": "same"}, run_dir=path)
            storage.observe({"event": "evaluation_start", "sequence": 0, "expected_images": 2})
            storage.observe({"event": "evaluation_complete", "sequence": 1, "is_best": True,
                             "params": {"alpha": 1.}, "score": .9}, output)
            with self.assertRaises(FileExistsError):
                RunStorage(root, {"input": "same"}, run_dir=path)
            with self.assertRaises(RuntimeError):
                RunStorage(root, {"input": "same"}, run_dir=path, resume=True)
            storage.close()
            with self.assertRaises(ValueError):
                RunStorage(root, {"input": "different"}, run_dir=path, resume=True)
            resumed = RunStorage(root, {"input": "same"}, run_dir=path, resume=True)
            restored = resumed.resume_output()
            self.assertTrue(torch.equal(restored.weights["w"], output.weights["w"]))
            self.assertTrue(torch.equal(restored.trace["basis"], torch.eye(2)))
            resumed.finish({"schema_version": 1, "status": "complete", "test": {}}, restored)
            with self.assertRaises(FileExistsError):
                resumed.finish({}, restored)
            resumed.close()
            with self.assertRaises(FileExistsError):
                RunStorage(root, {"input": "same"}, run_dir=path, resume=True)

    def test_tampered_best_weights_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            storage = RunStorage(root, {"a": 1})
            storage.observe({"sequence": 0, "event": "evaluation_complete", "is_best": True},
                            MergeOutput({"w": torch.tensor(1.)}))
            path = storage.path / storage.history[-1]["weights_file"]
            save_file({"w": torch.tensor(2.)}, str(path))
            with self.assertRaisesRegex(ValueError, "changed"):
                storage.resume_output()
            storage.close()

    def test_modified_validation_history_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as root:
            storage = RunStorage(root, {"a": 1})
            storage.observe({"sequence": 0, "event": "evaluation_start", "expected_images": 5})
            path = storage.path
            storage.close()
            journal = path / "trials.jsonl"
            event = json.loads(journal.read_text())
            event["expected_images"] = 1
            journal.write_text(json.dumps(event) + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                RunStorage(root, {"a": 1}, run_dir=path, resume=True)

    def test_comparison_does_not_mix_conditions_or_overwrite_csv(self):
        record = {
            "config": {"protocol": "common_fixed", "model": "M", "benchmark": "B", "runtime": {}},
            "input_identity": {"weights": "same"}, "validation_identity": {}, "test_identity": {},
            "method": {"id": "tsvm", "implementation": "ref"},
            "test": {"avg_top1": .8, "avg_normalized_top1": .9},
            "selected_params": {"alpha": 1}, "weights_fingerprint": "x",
        }
        other = {**record, "input_identity": {"weights": "different"}}
        with self.assertRaises(ValueError):
            check_comparability([record, other])
        rows = make_comparison_table([record, other], allow_mixed=True)
        self.assertNotEqual(rows[0]["comparison_group"], rows[1]["comparison_group"])
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "results.csv"
            export_csv(rows, destination)
            with self.assertRaises(FileExistsError):
                export_csv(rows, destination)

    def test_identity_is_order_independent_for_dicts(self):
        self.assertEqual(identity_hash({"a": 1, "b": 2}), identity_hash({"b": 2, "a": 1}))

    def test_comparison_separates_validation_budgets_but_allows_grid_and_bo(self):
        record = {"config": {"selection": {"mode": "grid", "max_validation_images": 100, "seed": 0}},
                  "input_identity": {}, "validation_identity": {}, "test_identity": {}}
        other = {**record, "config": {"selection": {"mode": "bayes", "max_validation_images": 100, "seed": 0}}}
        self.assertTrue(check_comparability([record, other])["comparable"])
        other["config"]["selection"]["max_validation_images"] = 10000
        with self.assertRaises(ValueError):
            check_comparability([record, other])


if __name__ == "__main__":
    unittest.main()
