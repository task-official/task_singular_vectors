"""Opt-in GPU reference and threshold tests; no downloads or real data."""
import contextlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.core.checkpoints import make_inputs
from src.core.types import MergeContext, RuntimeSpec
from src.methods import gse_tsv, tsvm, dc_merge, iso_cts
from tests.reference.gse_reference import _global_raw_energy_top_r_ranks, compute_component_selection_tsv_merge
from tests.reference.tsvm_reference import compute_and_sum_svd_mem_reduction
from tests.reference.dc_reference import dc_merge as dc_reference
from tests.reference.iso_reference import iso_cts as iso_reference


@unittest.skipUnless(os.environ.get("TSV_GPU_AUDIT_TESTS") == "1", "opt-in GPU audit")
class GpuAuditTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.fail("GPU audit requires explicit CUDA access")
        torch.set_num_threads(1)

    def test_raw_energy_boundary_matches_reference_gpu_arithmetic(self):
        generator = torch.Generator().manual_seed(101)
        chosen = None
        for count in (65, 257, 513, 1024):
            for _ in range(40):
                spectrum = torch.rand(count, generator=generator).sort(descending=True).values.cuda() * 1e-7
                cpu_total = float(spectrum.cpu().square().sum())
                gpu_total = float(spectrum.square().sum())
                if cpu_total != gpu_total:
                    chosen = spectrum, min(cpu_total, gpu_total)
                    break
            if chosen:
                break
        self.assertIsNotNone(chosen, "A CPU/CUDA reduction boundary must be exercised")
        spectrum, epsilon = chosen
        count = spectrum.numel()
        matrix = torch.diag(spectrum).cpu()
        inputs = make_inputs({"matrix": torch.zeros_like(matrix)}, {"A": {"matrix": matrix}})
        expected = _global_raw_energy_top_r_ranks([spectrum], epsilon)[0]
        # Exact analytical SVD of a sorted positive diagonal matrix isolates the
        # energy decision from unrelated cuSOLVER rounding.
        with patch("src.methods.gse_tsv.svd", return_value=(torch.eye(count, device="cuda"), spectrum, torch.eye(count, device="cuda"))):
            _, trace = gse_tsv.merge_layer(inputs, "matrix", gse_tsv.GseTsvParams(eps=epsilon), MergeContext(RuntimeSpec(device="cuda")))
        self.assertEqual(trace["ranks"]["A"], expected)

    def test_gpu_weights_match_frozen_references_for_rectangles_and_zeros(self):
        generator = torch.Generator().manual_seed(719)
        for shape in ((9, 9), (11, 7), (7, 11)):
            base = {"matrix": torch.randn(shape, generator=generator), "bias": torch.randn(shape[0], generator=generator)}
            tuned = {str(i): {name: value + torch.randn(value.shape, generator=generator) for name, value in base.items()} for i in range(3)}
            tuned["0"]["matrix"] = base["matrix"].clone()
            inputs = make_inputs(base, tuned)
            vectors = [SimpleNamespace(vector={name: inputs.delta(task, name) for name in inputs.parameter_order}) for task in inputs.task_order]
            config = SimpleNamespace(DATASETS=inputs.task_order, device="cuda", method=SimpleNamespace(eps=1e-12, common_space_fraction=.8))
            cases = ((tsvm, tsvm.TsvmParams(alpha=.7), lambda: compute_and_sum_svd_mem_reduction(vectors, config)),
                     (gse_tsv, gse_tsv.GseTsvParams(alpha=.7), lambda: compute_component_selection_tsv_merge(vectors, config, "global_raw_top_r", "gse", False)),
                     (dc_merge, dc_merge.DcMergeParams(alpha=.7), lambda: dc_reference(vectors, config)),
                     (iso_cts, iso_cts.IsoCtsParams(alpha=.7), lambda: iso_reference(vectors, config)))
            for module, parameters, reference in cases:
                with self.subTest(shape=shape, method=module.__name__):
                    with contextlib.redirect_stdout(io.StringIO()):
                        update = reference()
                    actual = module.merge(inputs, parameters, MergeContext(RuntimeSpec(device="cuda"))).weights
                    for name in inputs.parameter_order:
                        expected = (base[name].cuda()+.7*update[name].cuda()).cpu()
                        self.assertTrue(torch.equal(actual[name], expected), (module.__name__, name))

    def test_gpu_references_at_all_benchmark_task_counts(self):
        generator = torch.Generator().manual_seed(528)
        for count in (8, 14, 20):
            for shape in ((17, 31), (31, 17), (33, 33)):
                base = {"matrix": torch.zeros(shape)}
                tuned = {str(i): {"matrix": torch.randn(shape, generator=generator)} for i in range(count)}
                tuned["0"]["matrix"].zero_()
                tuned["1"]["matrix"] = torch.eye(*shape)  # Repeated singular values.
                inputs = make_inputs(base, tuned)
                vectors = [SimpleNamespace(vector={"matrix": inputs.delta(task, "matrix")}) for task in inputs.task_order]
                config = SimpleNamespace(DATASETS=inputs.task_order, device="cuda", method=SimpleNamespace(eps=1e-12, common_space_fraction=.8))
                cases = ((tsvm, tsvm.TsvmParams(alpha=.7), lambda: compute_and_sum_svd_mem_reduction(vectors, config)),
                         (gse_tsv, gse_tsv.GseTsvParams(alpha=.7), lambda: compute_component_selection_tsv_merge(vectors, config, "global_raw_top_r", "gse", False)),
                         (dc_merge, dc_merge.DcMergeParams(alpha=.7), lambda: dc_reference(vectors, config)),
                         (iso_cts, iso_cts.IsoCtsParams(alpha=.7), lambda: iso_reference(vectors, config)))
                for module, parameters, reference in cases:
                    with self.subTest(count=count, shape=shape, method=module.__name__):
                        with contextlib.redirect_stdout(io.StringIO()):
                            update = reference()
                        actual = module.merge(inputs, parameters, MergeContext(RuntimeSpec(device="cuda"))).weights
                        self.assertTrue(torch.equal(actual["matrix"], (.7 * update["matrix"].cuda()).cpu()))


if __name__ == "__main__":
    unittest.main()
