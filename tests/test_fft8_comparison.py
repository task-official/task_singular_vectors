"""Opt-in all-image data and the declared ten-method comparison conditions."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from PIL import Image
from torchvision.datasets import ImageFolder
import torch

from src.core.types import RuntimeSpec, TaskSpec
from src.datasets.adapter import DATASET_ADAPTERS, dataset_digest, inspect_task
from src.datasets.sun397_all_images import build_dataset
from src.experiments.analysis import check_comparability
from src.experiments.config import resolve_config
from src.experiments.registry import get_method


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN", "SUN397")
METHODS = ("zeroshot", "weight_average", "task_arithmetic", "ties", "consensus_ta",
           "tsvm", "iso_cts", "dc_merge", "sera", "gse_tsv")


def experiment(method):
    protocol = ("fft8_tuned_single" if method in ("zeroshot", "weight_average") else
                "fft8_tuned_sera" if method == "sera" else "fft8_tuned_alpha")
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        config = compose(config_name="experiment", overrides=[
            "model=vit_b32", "benchmark=fft8_all_images", f"method={method}",
            f"protocol={protocol}"])
    return resolve_config(config)


class AllImageDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for split in ("train", "val"):
            for name in ("b_bedroom", "a_art_school"):
                folder = self.root / "sun397" / split / name
                folder.mkdir(parents=True)
                for index in range(10):
                    image = Image.new("RGB", (8, 8), (200, 10, 20))
                    if index == 0:
                        image.save(folder / "0.GIF", save_all=True,
                                   append_images=[Image.new("RGB", (8, 8), (10, 200, 20))],
                                   duration=100, loop=0)
                    else:
                        image.save(folder / f"{index}.png")
                (folder / "notes.txt").write_text("not an image")

    def specification(self, split):
        return {"adapter": "sun397_all_images", "split": split,
                "path": "sun397/" + ("train" if split == "validation" else "val"),
                "val_fraction": .1, "max_val_samples": 5000, "seed": 0}

    def test_registered_factory_includes_gif_and_preserves_legacy_labels_and_files(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob("*") if p.is_file()}
        self.assertIs(DATASET_ADAPTERS["sun397_all_images"], build_dataset)
        all_images, names = build_dataset(self.specification("test"), self.root, None, RuntimeSpec())
        legacy = ImageFolder(self.root / "sun397/val")
        self.assertEqual(len(legacy), 18)
        self.assertEqual(len(all_images), 20)
        self.assertEqual(all_images.class_to_idx, legacy.class_to_idx)
        self.assertEqual(names, ("art school", "bedroom"))
        gifs = [i for i, (path, _) in enumerate(all_images.samples) if path.endswith(".GIF")]
        self.assertEqual(len(gifs), 2)
        for index in gifs:
            image, _ = all_images[index]
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((0, 0)), (200, 10, 20))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})

    def test_shared_partition_is_repeatable_and_inspection_keeps_split_identity(self):
        first, names = build_dataset(self.specification("validation"), self.root, None, RuntimeSpec())
        second, _ = build_dataset(self.specification("validation"), self.root, None, RuntimeSpec())
        self.assertEqual(len(first), 2)
        self.assertEqual(list(first.indices), list(second.indices))
        task = TaskSpec("SUN397", self.specification("validation"), self.specification("test"), "")
        validation = inspect_task(task, "validation", str(self.root), None, RuntimeSpec())
        test = inspect_task(task, "test", str(self.root), None, RuntimeSpec())
        self.assertEqual(validation.class_names, names)
        self.assertEqual(test.class_names, names)
        self.assertFalse(validation.membership.intersection(test.membership))
        self.assertEqual(validation.identity["membership_status"], "complete")
        self.assertEqual(test.identity["size"], 20)

    def test_gif_content_change_changes_dataset_identity(self):
        dataset, _ = build_dataset(self.specification("test"), self.root, None, RuntimeSpec())
        before = dataset_digest(dataset, self.root)
        Image.new("RGB", (8, 8), (40, 50, 60)).save(self.root / "sun397/val/a_art_school/0.GIF")
        self.assertNotEqual(before, dataset_digest(dataset, self.root))

    def test_invalid_split_and_class_directory_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "split"):
            build_dataset({**self.specification("test"), "split": "train"}, self.root, None, RuntimeSpec())
        (self.root / "sun397/val/a_art_school").rename(self.root / "sun397/val/art_school")
        with self.assertRaisesRegex(ValueError, "flattened class directories"):
            build_dataset(self.specification("test"), self.root, None, RuntimeSpec())

    def test_four_workers_preserve_transformed_images_labels_and_order(self):
        from torch.utils.data import DataLoader
        from torchvision.transforms import ToTensor

        dataset, _ = build_dataset(self.specification("test"), self.root, ToTensor(), RuntimeSpec())
        batches = []
        for workers in (0, 4):
            images, labels = [], []
            for batch_images, batch_labels in DataLoader(dataset, batch_size=3, shuffle=False,
                                                         num_workers=workers):
                images.append(batch_images)
                labels.append(batch_labels)
            batches.append((torch.cat(images), torch.cat(labels)))
        self.assertTrue(torch.equal(batches[0][0], batches[1][0]))
        self.assertTrue(torch.equal(batches[0][1], batches[1][1]))


class ComparisonConfigTests(unittest.TestCase):
    def test_ten_methods_resolve_validate_and_share_comparison_conditions(self):
        records = []
        for method_id in METHODS:
            with self.subTest(method=method_id):
                spec = experiment(method_id)
                self.assertEqual(tuple(t.id for t in spec.tasks), TASKS)
                self.assertEqual(spec.runtime.seed, 0)
                self.assertEqual(spec.selection.max_validation_images, 8000000)
                self.assertEqual(spec.selection.objective, "macro_normalized_top1")
                method = get_method(method_id)
                for override in spec.selection.candidates:
                    method.parameters({**spec.params, **override})
                if method_id in ("zeroshot", "weight_average"):
                    self.assertEqual(spec.selection.candidates, ({},))
                    self.assertEqual(spec.selection.mode, "grid")
                else:
                    self.assertEqual([p["alpha"] for p in spec.selection.candidates],
                                     [i / 10 for i in range(31)])
                records.append({"config": asdict(spec), "input_identity": {"same": "inputs"},
                                "validation_identity": {"same": "validation"},
                                "test_identity": {"same": "test"}})
        self.assertTrue(check_comparability(records)["comparable"])

    def test_budget_covers_all_proposed_sera_and_consensus_evaluations(self):
        validation_images = 21059
        sera = experiment("sera")
        self.assertEqual(sera.selection.mode, "bayes")
        self.assertEqual(sera.selection.initial_energies, (.45, .65, .85))
        self.assertEqual(sera.selection.energy_bounds, (.1, .99))
        self.assertEqual(sera.selection.max_trials, 12)
        self.assertEqual(sera.selection.ei_tolerance, .01)
        needed = sera.selection.max_trials * len(sera.selection.candidates) * validation_images
        self.assertLessEqual(needed, sera.selection.max_validation_images)
        consensus = experiment("consensus_ta")
        needed = (len(consensus.params["mask_lambdas"]) + len(consensus.selection.candidates)) * validation_images
        self.assertLessEqual(needed, consensus.selection.max_validation_images)

    def test_legacy_benchmark_and_other_tasks_keep_their_adapters(self):
        new = experiment("tsvm")
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
            old = resolve_config(compose(config_name="experiment", overrides=["benchmark=fft8"]))
        self.assertEqual(tuple(t.id for t in old.tasks), TASKS)
        self.assertTrue(all(t.validation["adapter"] == "legacy" and t.test["adapter"] == "legacy"
                            for t in old.tasks))
        for task in new.tasks:
            if task.id == "SUN397":
                self.assertEqual(task.validation["adapter"], "sun397_all_images")
                self.assertEqual(task.test["adapter"], "sun397_all_images")
            else:
                self.assertEqual(task, next(t for t in old.tasks if t.id == task.id))


class OpenAiArchitectureTests(unittest.TestCase):
    def test_offline_adapter_matches_independent_openai_constructor(self):
        from open_clip.model import CLIP, build_model_from_openai_state_dict
        from src.models.adapter import OpenClipAdapter

        # A small real transformer exercises the production OpenCLIP factory.
        # The independent reference infers its architecture from an OpenAI-
        # format state dictionary and uses that constructor's QuickGELU default.
        config = {"embed_dim": 32,
                  "vision_cfg": {"image_size": 32, "layers": 1, "width": 64, "patch_size": 16},
                  "text_cfg": {"context_length": 8, "vocab_size": 64, "width": 64, "heads": 1, "layers": 1}}
        source = CLIP(**config, quick_gelu=True)
        weights = source.state_dict()
        reference = build_model_from_openai_state_dict(dict(weights), cast_dtype=torch.float32).float()
        # The official builder briefly casts its weights to FP16. Restore the
        # same FP32 source values used by the production adapter before comparing.
        reference.load_state_dict(weights)
        spec = SimpleNamespace(model="ViT-B-32", pretrained_tag="openai", runtime=RuntimeSpec(), tasks=())
        encoded_weights = {"model." + key: value for key, value in weights.items()}
        with patch("open_clip.factory.get_model_config", return_value=config):
            adapter = OpenClipAdapter(spec, tuple(encoded_weights))
        adapter.load_weights(encoded_weights)
        self.assertEqual(adapter.identity["activation"], "quick_gelu")
        images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(21))
        with torch.no_grad():
            self.assertTrue(torch.equal(reference.encode_image(images), adapter.encoder(images)))


if __name__ == "__main__":
    unittest.main()
