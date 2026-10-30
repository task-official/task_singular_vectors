"""Old numerical entry points delegate to the canonical calculation."""
import contextlib
import io
from types import SimpleNamespace
import unittest

import torch

from src.compat.cli import translate_legacy_config
from src.utils.TSVM_utils import compute_and_sum_svd_mem_reduction, compute_component_selection_tsv_merge
from tests.reference.experimental_reference import legacy_variants
from tests.reference.tsvm_reference import compute_and_sum_svd_mem_reduction as reference


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def vectors(self, dtype=torch.float32):
        generator = torch.Generator().manual_seed(9)
        return [SimpleNamespace(vector={
            "matrix": torch.randn(8, 8, generator=generator, dtype=dtype) if index else torch.zeros(8, 8, dtype=dtype),
            "bias": torch.randn(8, generator=generator, dtype=dtype),
        }) for index in range(3)]

    def test_old_tsvm_api_matches_frozen_reference_and_diagnostic_entry(self):
        config = SimpleNamespace(DATASETS=("a", "b", "c"), device="cpu")
        for dtype in (torch.float32, torch.float64):
            vectors = self.vectors(dtype)
            with contextlib.redirect_stdout(io.StringIO()):
                expected = reference(vectors, config)
            actual, diagnostics = compute_and_sum_svd_mem_reduction(vectors, config, True)
            reported, _ = compute_component_selection_tsv_merge(vectors, config, "tsvm", "TSVM", True)
            for name in expected:
                self.assertTrue(torch.equal(expected[name], actual[name]), name)
                self.assertTrue(torch.equal(actual[name], reported[name]), name)
            self.assertEqual(diagnostics["method"], "TSVM")
            self.assertEqual(diagnostics["sti"], {"status": "not_collected"})
            self.assertIn("reconstruction", diagnostics)
            layer = diagnostics["observed_trace"]["layers"]["matrix"]
            for task in config.DATASETS:
                expected_energy = layer["selected_energy_sum"][task] / layer["total_energy"][task] if layer["total_energy"][task] else 0
                self.assertEqual(diagnostics["selection_energy"]["task_scores"][task], expected_energy)

    def test_adaptive_diagnostics_use_classification_rank_allocator(self):
        config = SimpleNamespace(DATASETS=("a", "b", "c"), device="cpu",
                                 method=SimpleNamespace(energy_threshold=.95, eps=1e-12))
        vectors = self.vectors()
        with contextlib.redirect_stdout(io.StringIO()):
            expected, diag = legacy_variants.compute_adaptive_tsv_merge(vectors, config, True)
            actual, actual_diag = compute_component_selection_tsv_merge(vectors, config, "adaptive", "AdaptiveTSV", True)
        for name in expected:
            self.assertTrue(torch.equal(expected[name], actual[name]))
        for name, layer in diag["rank"]["per_layer"].items():
            self.assertEqual(layer["ranks"], actual_diag["rank"]["per_layer"][name]["ranks"])

    def test_original_readme_remains_byte_identical(self):
        from pathlib import Path
        import hashlib
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(hashlib.sha256((root / "README.original.md").read_bytes()).hexdigest(),
                         "cf8295325132cc98988ed88fee24ca40ad1cc1e1a786441256a13550e318cf22")


if __name__ == "__main__":
    unittest.main()
