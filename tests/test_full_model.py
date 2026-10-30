"""Opt-in, offline whole-encoder regression with the eight supplied B/32 tasks.

Run prepare.py first. No dataset, download or training is performed. The test
compares every weight, then all supplied heads' logits and predictions on fixed
inputs, against the frozen original arithmetic/TSV computations.
"""
import contextlib
import gc
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from src.core.checkpoints import make_inputs, open_prepared, tensor_fingerprint
from src.core.types import MergeContext, RuntimeSpec
from src.experiments.registry import get_method
from tests.reference.tsvm_reference import compute_and_sum_svd_mem_reduction as tsvm_reference


@unittest.skipUnless(os.environ.get("TSV_FULL_MODEL_TESTS") == "1", "opt-in whole-model regression")
class WholeModelTests(unittest.TestCase):
    def test_four_baselines_every_weight_and_all_head_logits(self):
        if not torch.cuda.is_available():
            self.fail("Whole-model verification requires explicit CUDA access")
        from hydra import compose, initialize_config_dir
        from src.experiments.config import resolve_config
        from src.models.adapter import OpenClipAdapter
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root / "config"), version_base="1.3"):
            spec = resolve_config(compose(config_name="experiment"))
        base = open_prepared(spec.base_path, spec.runtime.cache_dir)
        tuned = {task.id: open_prepared(spec.checkpoint_paths[task.id], spec.runtime.cache_dir) for task in spec.tasks}
        torch.set_num_threads(4)
        inputs = make_inputs(base, tuned)
        adapter = OpenClipAdapter(spec, inputs.parameter_order)
        images = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(360)).cuda()
        runtime = RuntimeSpec(device="cuda", trace="none")
        names = [name for name in inputs.parameter_order if base.tensor(name).is_floating_point()]
        # The reference stacks flattened deltas on CPU. Keep one original global
        # stack so the sum's CPU vector/scalar reduction regions stay unchanged.
        count = sum(base.tensor(name).numel() for name in names)
        stacked = torch.empty((len(inputs.task_order), count))
        offset = 0
        for name in names:
            original = base.tensor(name)
            width = original.numel()
            for row, task in enumerate(inputs.task_order):
                stacked[row, offset:offset+width] = (tuned[task].tensor(name) - original).reshape(-1)
            offset += width
        summed = stacked.sum(dim=0)
        del stacked
        for method_id, alpha in (("zeroshot", 0.), ("weight_average", 1/8), ("task_arithmetic", .5), ("tsvm", .7)):
            print(f"Whole-model reference: {method_id}", flush=True)
            method = get_method(method_id)
            params = {} if method_id in ("zeroshot", "weight_average") else {"alpha": alpha}
            actual = method.merge(inputs, method.parameters(params), MergeContext(runtime)).weights
            expected = {}
            offset = 0
            for name in inputs.parameter_order:
                original = base.tensor(name)
                if not original.is_floating_point() or method_id == "zeroshot":
                    expected[name] = original.clone()
                    if original.is_floating_point():
                        offset += original.numel()
                    continue
                if method_id == "tsvm":
                    vectors = [SimpleNamespace(vector={name: tuned[task].tensor(name)-original}) for task in inputs.task_order]
                    with contextlib.redirect_stdout(io.StringIO()):
                        update = tsvm_reference(vectors, SimpleNamespace(DATASETS=inputs.task_order, device="cuda"))[name]
                    expected[name] = (original.cuda()+alpha*update.cuda()).cpu()
                    del vectors, update
                else:
                    update = summed[offset:offset+original.numel()].view(original.shape)
                    expected[name] = (original.cuda()+alpha*update.cuda()).cpu()
                offset += original.numel()
                self.assertTrue(torch.equal(expected[name], actual[name]), (method_id, name))
            self.assertEqual(tensor_fingerprint(expected), tensor_fingerprint(actual), method_id)
            adapter.load_weights(expected)
            with torch.no_grad():
                reference_logits = {task: adapter.logits(images, task).cpu() for task in inputs.task_order}
            adapter.load_weights(actual)
            with torch.no_grad():
                for task in inputs.task_order:
                    logits = adapter.logits(images, task).cpu()
                    self.assertTrue(torch.equal(reference_logits[task], logits), (method_id, task))
                    self.assertTrue(torch.equal(reference_logits[task].argmax(1), logits.argmax(1)))
            print(f"Verified {method_id}: {len(actual)} tensors; 8 heads; exact logits/predictions", flush=True)
            del actual, expected, reference_logits
            gc.collect()
            torch.cuda.empty_cache()
