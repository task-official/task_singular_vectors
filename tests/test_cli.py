"""Exercise actual CLI parsing/conversion/export in isolated temporary folders."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import load_file
import torch

from src.compat.cli import translate_legacy_config
from src.experiments.config import resolve_config

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("weight_average", "task_arithmetic", "ties", "consensus_ta", "tsvm",
           "iso_cts", "dc_merge", "sera", "gse_tsv")


def compose_experiment(overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        return compose(config_name="experiment", overrides=list(overrides))


class CliTests(unittest.TestCase):
    def command(self, script, arguments, directory, success=True):
        result = subprocess.run([sys.executable, "-B", str(ROOT / script), *arguments],
                                cwd=directory, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_all_81_configs_resolve_and_validate_declared_method_parameters(self):
        from src.experiments.registry import get_method
        for model in ("vit_b32", "vit_b16", "vit_l14"):
            for benchmark in ("fft8", "fft14", "fft20"):
                for method in METHODS:
                    with self.subTest(model=model, benchmark=benchmark, method=method):
                        spec = resolve_config(compose_experiment([
                            f"model={model}", f"benchmark={benchmark}", f"method={method}"]))
                        get_method(method).parameters(spec.params)
                        self.assertEqual(len(spec.tasks), int(benchmark.removeprefix("fft")))

    def test_tuned_average_and_zeroshot_do_not_invent_alpha(self):
        from src.experiments.registry import get_method
        for method in ("weight_average", "zeroshot"):
            spec = resolve_config(compose_experiment([
                f"method={method}", "protocol=common_tuned",
                "protocol.selection.max_validation_images=100"]))
            self.assertEqual(spec.selection.candidates, ({},))
            for candidate in spec.selection.candidates:
                get_method(method).parameters({**spec.params, **candidate})

    def test_resume_argument_is_execution_control_not_scientific_identity(self):
        first = resolve_config(compose_experiment())
        second = resolve_config(compose_experiment(["resume_from=/tmp/example-run"]))
        self.assertEqual(first, second)

    def test_run_entry_forwards_resume_and_prints_published_result(self):
        import run
        config = compose_experiment(["resume_from=/tmp/example-run"])
        result = {"run_dir": "/tmp/example-run", "test": {"avg_top1": .5}}
        with patch("src.experiments.runner.run_experiment", return_value=result) as execute:
            self.assertIs(run.main.__wrapped__(config), result)
        self.assertEqual(execute.call_args.kwargs, {"run_dir": "/tmp/example-run", "resume": True})

    def test_actual_hydra_cli_accepts_new_groups_and_resolves_search_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.command("run.py", ["--cfg", "job", "--resolve", "model=vit_l14",
                                  "benchmark=fft20", "method=weight_average", "protocol=common_tuned",
                                  "protocol.selection.max_validation_images=100"], directory)
            data = OmegaConf.to_container(OmegaConf.create(result.stdout), resolve=True)
            self.assertEqual(data["model"]["name"], "ViT-L-14")
            self.assertEqual(data["protocol"]["selection"]["candidates"], [{}])
            self.assertEqual(len(data["benchmark"]["task_ids"]), 20)

    def test_actual_prepare_cli_preserves_inputs_and_verifies_small_tensor_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "models"
            model_root = checkpoint_root / "ViT-B-32"
            (model_root / "MNISTVal").mkdir(parents=True)
            (model_root / "AVal").mkdir()
            base = model_root / "MNISTVal/nonlinear_zeroshot.pt"
            tuned = model_root / "AVal/nonlinear_finetuned.pt"
            torch.save({"w": torch.zeros(2, 2)}, base)
            torch.save({"w": torch.ones(2, 2)}, tuned)
            (model_root / "head_AVal.pt").write_bytes(b"prepared head fixture")
            original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (base, tuned)}
            arguments = ["benchmark.task_ids=[A]", "benchmark.name=synthetic_cli",
                         f"inputs.checkpoint_root={checkpoint_root}", f"runtime.cache_dir={root / 'cache'}",
                         "runtime.device=cpu"]
            self.command("prepare.py", arguments, directory)
            converted = list((root / "cache/checkpoints").glob("*.safetensors"))
            self.assertEqual(len(converted), 2)
            self.assertEqual(sorted(load_file(str(path))["w"].sum().item() for path in converted), [0., 4.])
            self.command("prepare.py", arguments, directory)
            for path, digest in original.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_actual_compare_cli_checks_identity_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"schema_version": 1, "status": "complete", "source_type": "local",
                      "config": {"protocol": "common_fixed", "model": "synthetic", "benchmark": "B"},
                      "input_identity": {"weights": "same"}, "validation_identity": {}, "test_identity": {},
                      "method": {"id": "tsvm", "implementation": "test"},
                      "selected_params": {"alpha": 1}, "weights_fingerprint": "fixture",
                      "test": {"avg_top1": .8, "avg_normalized_top1": .9}}
            first, second = root / "one.json", root / "two.json"
            first.write_text(json.dumps(record))
            second.write_text(json.dumps({**record, "method": {"id": "gse_tsv", "implementation": "test"}}))
            output = root / "comparison.csv"
            self.command("compare.py", [str(first), str(second), "--output", str(output)], directory)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["method"] for row in rows], ["tsvm", "gse_tsv"])
            self.command("compare.py", [str(first), "--output", str(output)], directory, success=False)
            altered = {**record, "test_identity": {"images": "different"}}
            second.write_text(json.dumps(altered))
            self.command("compare.py", [str(first), str(second), "--output", str(root / "mixed.csv")], directory, success=False)
            self.assertFalse((root / "mixed.csv").exists())

    def test_legacy_aliases_preserve_reference_coefficient_grids(self):
        for name, expected in (("TSVM", 3.0), ("sum", 1.0)):
            config = OmegaConf.create({
                "method": {"name": name, "k": 100}, "model": "ViT-B-32", "DATASETS": "",
                "num_tasks": 8, "specify_lambda": "None", "n_eval_points": 31,
                "model_location": "models/checkpoints", "data_location": "datasets",
                "device": "cpu", "batch_size": 32,
            })
            spec = translate_legacy_config(config)
            self.assertEqual(len(spec.selection.candidates), 30)
            self.assertEqual(spec.selection.candidates[-1], {"alpha": expected})
            self.assertEqual(spec.selection.mode, "grid")


if __name__ == "__main__":
    unittest.main()
