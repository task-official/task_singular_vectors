"""Opt-in local-artifact regressions; never acquire weights or data."""
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.core.checkpoints import make_inputs
from src.core.types import MergeContext, RuntimeSpec, TaskSpec
from src.methods.tsvm import merge, TsvmParams
from tests.reference.tsvm_reference import compute_and_sum_svd_mem_reduction as reference


@unittest.skipUnless(os.environ.get("TSV_REAL_CHECKPOINT_TESTS") == "1", "opt-in local checkpoint regression")
class RealCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path(os.environ.get("TSV_CHECKPOINT_ROOT", "models/checkpoints")) / "ViT-B-32"
        self.base_path = self.root / "MNISTVal/nonlinear_zeroshot.pt"
        if not self.base_path.is_file():
            self.skipTest("Local ViT-B-32 baseline checkpoint is unavailable")

    def test_eight_task_representative_layer_matches_reference(self):
        tasks = ("Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN", "SUN397")
        paths = {task: self.root / f"{task}Val/nonlinear_finetuned.pt" for task in tasks}
        if not all(path.is_file() for path in paths.values()):
            self.skipTest("Eight local fine-tuned checkpoints are required")
        names = ("model.visual.transformer.resblocks.0.attn.in_proj_weight",
                 "model.visual.transformer.resblocks.0.ln_1.weight")
        checkpoint = torch.load(self.base_path, map_location="cpu")
        base = {name: checkpoint[name].clone() for name in names}
        del checkpoint
        tuned = {}
        for task, path in paths.items():
            checkpoint = torch.load(path, map_location="cpu")
            tuned[task] = {name: checkpoint[name].clone() for name in names}
            del checkpoint
        vectors = [SimpleNamespace(vector={name: tuned[task][name] - base[name] for name in names})
                   for task in tasks]
        with contextlib.redirect_stdout(io.StringIO()):
            expected = reference(vectors, SimpleNamespace(DATASETS=tasks, device="cpu"))
        actual = merge(make_inputs(base, tuned, tasks), TsvmParams(), MergeContext(RuntimeSpec())).weights
        for name in names:
            self.assertTrue(torch.equal(actual[name], base[name] + expected[name]), name)

    def test_supplied_mnist_head_and_preprocessing_match_legacy_path(self):
        import open_clip
        from src.models.adapter import OpenClipAdapter
        from src.models.modeling import ImageEncoder, ImageClassifier, ClassificationHead
        head_path = self.root / "head_MNISTVal.pt"
        if not head_path.is_file():
            self.skipTest("Local supplied MNIST head is required")
        weights = torch.load(self.base_path, map_location="cpu")
        factory = open_clip.create_model_and_transforms

        def offline(*args, **kwargs):
            kwargs["pretrained"] = None
            return factory(*args, **kwargs)

        with patch("open_clip.create_model_and_transforms", side_effect=offline):
            legacy_encoder = ImageEncoder("ViT-B-32")
        legacy_encoder.load_state_dict(weights)
        legacy = ImageClassifier(legacy_encoder, ClassificationHead.load(str(head_path))).eval()
        spec = SimpleNamespace(model="ViT-B-32", pretrained_tag="openai", runtime=RuntimeSpec(),
                               tasks=(TaskSpec("MNIST", {}, {}, str(head_path)),))
        adapter = OpenClipAdapter(spec, tuple(weights))
        adapter.load_weights(weights)
        self.assertEqual(repr(legacy.val_preprocess), repr(adapter.preprocess))
        images = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(36))
        with torch.no_grad():
            old, new = legacy(images), adapter.logits(images, "MNIST")
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(old.argmax(1), new.argmax(1)))


if __name__ == "__main__":
    unittest.main()
