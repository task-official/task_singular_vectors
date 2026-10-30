import json
import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.core.checkpoints import MemoryTensorSource, make_inputs, open_prepared, prepare_checkpoint
from src.experiments.config import resolve_config


class InputTests(unittest.TestCase):
    def test_preparation_round_trip_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            original = {"w": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                        "counter": torch.tensor(2, dtype=torch.int64)}
            torch.save(original, source)
            before = source.read_bytes()
            converted = prepare_checkpoint(source, root / "cache")
            reader = open_prepared(source, root / "cache")
            for name, value in original.items():
                self.assertTrue(torch.equal(value, reader.tensor(name)))
            self.assertEqual(before, source.read_bytes())
            self.assertEqual(converted, prepare_checkpoint(source, root / "cache"))
            metadata = converted.with_suffix(".json")
            data = json.loads(metadata.read_text())
            data["converted_sha256"] = "wrong"
            metadata.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                open_prepared(source, root / "cache")

    def test_missing_prepare_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            torch.save({"w": torch.ones(2)}, source)
            with self.assertRaisesRegex(FileNotFoundError, "prepare.py"):
                open_prepared(source, directory)

    def test_shape_dtype_keys_and_buffers(self):
        base = {"w": torch.ones(2, 3), "count": torch.tensor(1)}
        for invalid in ({"w": torch.ones(3, 2), "count": torch.tensor(1)},
                        {"w": torch.ones(2, 3).double(), "count": torch.tensor(1)},
                        {"w": torch.ones(2, 3), "count": torch.tensor(2)},
                        {"w": torch.ones(2, 3)}):
            with self.assertRaises(ValueError):
                make_inputs(base, {"a": invalid})
        with self.assertRaises(ValueError):
            make_inputs(base, {"a": base}, ("a", "a"))

    def test_identity_binds_order_and_values(self):
        base = {"w": torch.zeros(2)}
        tuned = {"a": {"w": torch.ones(2)}, "b": {"w": torch.ones(2) * 2}}
        first = make_inputs(base, tuned, ("a", "b"))
        reverse = make_inputs(base, tuned, ("b", "a"))
        self.assertNotEqual(first.identity, reverse.identity)

    def config(self, *overrides):
        config_dir = str(Path("config").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            config = compose(config_name="experiment", overrides=list(overrides))
        return resolve_config(config)

    def test_all_81_configs_resolve(self):
        methods = ("weight_average", "task_arithmetic", "ties", "consensus_ta", "tsvm",
                   "iso_cts", "dc_merge", "sera", "gse_tsv")
        for model in ("vit_b32", "vit_b16", "vit_l14"):
            for number in (8, 14, 20):
                for method in methods:
                    spec = self.config(f"model={model}", f"benchmark=fft{number}", f"method={method}")
                    self.assertEqual(len(spec.tasks), number)
                    self.assertEqual(spec.method, method)

    def test_tuned_budget_required(self):
        with self.assertRaises(Exception):
            self.config("protocol=common_tuned")
        spec = self.config("protocol=common_tuned", "protocol.selection.max_validation_images=1000")
        self.assertEqual(spec.selection.max_validation_images, 1000)

    def test_average_has_no_meaningless_alpha_search(self):
        spec = self.config("method=weight_average", "protocol=common_tuned",
                           "protocol.selection.max_validation_images=1000")
        self.assertEqual(spec.selection.candidates, ({},))


if __name__ == "__main__":
    unittest.main()
