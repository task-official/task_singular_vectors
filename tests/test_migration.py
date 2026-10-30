"""Acceptance checks for the completed migration, beyond individual kernels."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from src.compat.cli import translate_legacy_config
from src.core.checkpoints import make_inputs, prepare_checkpoint, prepared_path, open_prepared, tensor_fingerprint
from src.core.types import MergeContext, RuntimeSpec
from src.experiments.registry import get_method
from tests.reference.experimental_reference import legacy_variants as old

ROOT = Path(__file__).resolve().parents[1]


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_research_variants_match_frozen_pre_migration_weights(self):
        for shape in ((8, 8), (11, 6), (6, 11)):
            generator = torch.Generator().manual_seed(37)
            vectors = [SimpleNamespace(vector={"matrix": torch.randn(*shape, generator=generator),
                                                "bias": torch.randn(shape[0], generator=generator)}) for _ in range(3)]
            vectors[0].vector["matrix"].zero_()
            cfg = SimpleNamespace(DATASETS=("a", "b", "c"), device="cpu",
                                  method=SimpleNamespace(energy_threshold=.95, eps=1e-12))
            base = {name: torch.zeros_like(value) for name, value in vectors[0].vector.items()}
            inputs = make_inputs(base, {task: vector.vector for task, vector in zip(cfg.DATASETS, vectors)})
            cases = [
                ("adaptive_tsv", {}, lambda: old.compute_adaptive_tsv_merge(vectors, cfg)),
                ("subspace_tsv", {"weighted": False}, lambda: old.compute_adaptive_tsv_subspace(vectors, cfg, False)),
                ("subspace_tsv", {"weighted": True}, lambda: old.compute_adaptive_tsv_subspace(vectors, cfg, True)),
                ("eigen_tsv", {}, lambda: old.compute_and_sum_svd_mem_reduction_2(vectors, cfg)),
            ]
            for mode, rule in (("global_top_r", "normalized_top_r"), ("per_task_95", "per_task_energy"), ("full", "full")):
                cases.append(("component_selection", {"selection": rule},
                              lambda mode=mode: old.compute_component_selection_tsv_merge(vectors, cfg, mode, mode)))
            for method_id, values, reference in cases:
                with self.subTest(shape=shape, method=method_id, params=values):
                    with contextlib.redirect_stdout(io.StringIO()):
                        expected = reference()
                    method = get_method(method_id)
                    for trace in ("none", "full"):
                        actual = method.merge(inputs, method.parameters(values), MergeContext(RuntimeSpec(trace=trace))).weights
                        for name in expected:
                            self.assertTrue(torch.equal(actual[name], expected[name]), (method_id, shape, name, trace))

    def test_dummy_uses_same_random_sequence_and_individual_is_explicit(self):
        inputs = make_inputs({"matrix": torch.zeros(8, 8)}, {"a": {"matrix": torch.eye(8)}})
        vectors = [SimpleNamespace(vector={"matrix": torch.eye(8)})]
        cfg = SimpleNamespace(DATASETS=("a",), device="cpu")
        torch.manual_seed(93)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = old.compute_and_sum_svd_mem_reduction_dummy(vectors, cfg)
        torch.manual_seed(93)
        method = get_method("dummy")
        actual = method.merge(inputs, method.parameters({}), MergeContext(RuntimeSpec(seed=93))).weights
        self.assertTrue(torch.equal(actual["matrix"], expected["matrix"]))
        one = get_method("individual")
        output = one.merge(inputs, one.parameters({}), MergeContext(RuntimeSpec()))
        self.assertTrue(torch.equal(output.weights["matrix"], torch.eye(8)))

    def test_preparation_recovers_both_incomplete_publication_states(self):
        for orphan in ("tensors", "metadata"):
            with self.subTest(orphan=orphan), tempfile.TemporaryDirectory() as root:
                source = Path(root) / "source.pt"
                weights = {"w": torch.tensor([1., 2.])}
                torch.save(weights, source)
                path = prepare_checkpoint(source, Path(root) / "cache")
                if orphan == "tensors":
                    path.with_suffix(".json").unlink()
                else:
                    path.unlink()
                self.assertEqual(prepare_checkpoint(source, Path(root) / "cache"), path)
                self.assertTrue(torch.equal(open_prepared(source, Path(root) / "cache").tensor("w"), weights["w"]))
                self.assertTrue(torch.equal(torch.load(source)["w"], weights["w"]))

    def test_preparation_does_not_adopt_corrupted_orphan(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.pt"
            torch.save({"w": torch.tensor([1.])}, source)
            cache = Path(root) / "cache"
            path = prepared_path(source, cache)
            path.parent.mkdir(parents=True)
            save_file({"w": torch.tensor([2.])}, str(path))
            with self.assertRaisesRegex(ValueError, "differs from source"):
                prepare_checkpoint(source, cache)

    def test_legacy_settings_are_forwarded_and_special_modes_do_not_fall_back(self):
        cfg = OmegaConf.create({"method": {"name": "AdaptiveTSV", "k": 100, "eps": 1e-12},
              "model": "ViT-B-32", "DATASETS": "", "num_tasks": 8, "specify_lambda": "None", "n_eval_points": 31,
              "model_location": "models/checkpoints", "data_location": "datasets", "device": "cpu", "batch_size": 32,
              "save_subfolder": "my-results", "cache_dir": "my-cache"})
        spec = translate_legacy_config(cfg)
        self.assertEqual(spec.runtime.cache_dir, "my-cache")
        self.assertEqual(spec.runtime.output_dir, "results/my-results")
        self.assertEqual(spec.method, "adaptive_tsv")
        for method in ("tall_mask", "mag_masking", "TSVC"):
            cfg.method.name = method
            with self.assertRaisesRegex(ValueError, "archive/legacy_code"):
                translate_legacy_config(cfg)

    def test_only_the_canonical_execution_path_is_importable(self):
        self.assertFalse((ROOT / "src/compat/legacy_cli.py").exists())
        self.assertFalse((ROOT / "src/eval").exists())
        self.assertFalse((ROOT / "src/methods/experimental/legacy_variants.py").exists())
        for path in (ROOT / "src/methods").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertFalse(module.startswith(("src.compat", "src.experiments", "src.evaluation", "hydra", "wandb")), (path, module))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name, ("hydra", "wandb"), path)

    def test_custom_factory_code_is_part_of_dataset_implementation_identity(self):
        from src.datasets.adapter import implementation_digest
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "factory.py"
            path.write_text("def factory():\n    return 1\n")
            import importlib.util
            spec = importlib.util.spec_from_file_location("temporary_factory", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            before = implementation_digest(module.factory)
            path.write_text("def factory():\n    return 2\n")
            self.assertNotEqual(before, implementation_digest(module.factory))

    def test_rank_study_and_individual_entrypoints_use_single_task_runs(self):
        from src.core.types import ExperimentSpec, SelectionSpec, TaskSpec
        from src.evaluation.single_task import run_individuals
        from src.experiments.rank_study import run_rank_study
        tasks = tuple(TaskSpec(task, {}, {}, "head") for task in ("A", "B"))
        spec = ExperimentSpec("synthetic", tasks, {"A": "a", "B": "b"}, "base", "data", "tsvm", {}, SelectionSpec(), RuntimeSpec())
        calls = []
        def run(one, **kwargs):
            calls.append(one)
            return {"run_dir": "synthetic"}
        run_individuals(spec, run=run)
        self.assertEqual([one.tasks[0].id for one in calls], ["A", "B"])
        self.assertTrue(all(one.method == "individual" and len(one.tasks) == 1 for one in calls))
        calls.clear()
        run_rank_study(spec, (.1, .5), run=run)
        self.assertEqual([one.params["rank_fraction"] for one in calls], [.1, .1, .5, .5])

    def test_training_bridge_evaluates_existing_encoder_without_rebuilding(self):
        from src.compat.training import eval_single_dataset
        from torch.utils.data import TensorDataset
        class Encoder(torch.nn.Identity):
            val_preprocess = None
        with tempfile.TemporaryDirectory() as root:
            torch.save({"weight": torch.eye(2), "normalize": False}, Path(root) / "head_AVal.pt")
            data = TensorDataset(torch.eye(2), torch.tensor([0, 1]))
            args = SimpleNamespace(save_dir=root, data_location=root, device="cpu", batch_size=2, num_workers=0)
            with patch.dict("src.datasets.adapter.DATASET_ADAPTERS", {"legacy": lambda *a: (data, ("a", "b"))}):
                result = eval_single_dataset(Encoder(), "AVal", args)
            self.assertEqual(result, {"top1": 1.0})

    def test_legacy_file_splits_cannot_silently_change_class_label_mapping(self):
        from src.datasets.adapter import _legacy
        from torch.utils.data import TensorDataset
        data = TensorDataset(torch.eye(2), torch.tensor([0, 1]))
        data.class_to_idx = {"a": 0, "b": 1}
        train = SimpleNamespace(class_to_idx={"a": 1, "b": 0})
        bundle = SimpleNamespace(train_dataset=train, test_dataset=data, classnames=("a", "b"))
        with patch("src.datasets.registry.get_dataset", return_value=bundle):
            with self.assertRaisesRegex(ValueError, "class-to-label"):
                _legacy({"name": "EuroSAT", "split": "test"}, Path("unused"), None, RuntimeSpec())

    def test_research_configs_and_legacy_cli_resolve_from_the_final_tree(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
            for path in (ROOT / "config/method/experimental").glob("*.yaml"):
                spec = __import__("src.experiments.config", fromlist=["resolve_config"]).resolve_config(
                    compose(config_name="experiment", overrides=[f"method=experimental/{path.stem}"]))
                get_method(spec.method).parameters(spec.params)
        with initialize_config_dir(config_dir=str(ROOT / "config/legacy"), version_base="1.3"):
            for name in ("zeroshot", "average", "sum", "TSVM", "ties", "consensus", "AdaptiveTSV", "SubspaceTSV", "AdaptiveSubspaceTSV", "GlobalEnergyTopR", "GlobalRawEnergyTopR", "PerTask95FullConcat", "FullComponentFullConcat", "TSVM_2", "dummy"):
                cfg = compose(config_name="config", overrides=[f"method={name}", "device=cpu"])
                spec = translate_legacy_config(cfg)
                get_method(spec.method).parameters(spec.params)

    def test_saved_result_plotting_and_historical_import_are_read_only(self):
        from src.compat.results import legacy_view
        from src.experiments.analysis import make_comparison_table
        from src.experiments.plotting import plot_comparison
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "old.json"
            original = '{"test":{"avg_top1":0.8,"avg_normalized_top1":1.1,"A:top1":0.8}}'
            path.write_text(original)
            record = legacy_view(path, "old-method")
            table = make_comparison_table([record])
            images = plot_comparison([record], table, Path(root) / "figures")
            self.assertEqual(len(images), 6)
            self.assertTrue(all(Path(p).is_file() for p in images))
            self.assertEqual(path.read_text(), original)
            with self.assertRaises(FileExistsError):
                plot_comparison([record], table, Path(root) / "figures")


if __name__ == "__main__":
    unittest.main()
