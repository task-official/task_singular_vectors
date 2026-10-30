"""Numerical characterization, algorithm edges, cache and observer invariance."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.core.checkpoints import make_inputs
from src.core.numerics import rank_for_energy
from src.core.types import MergeContext, RuntimeSpec
from src.methods import arithmetic, consensus, dc_merge, gse_tsv, iso_cts, sera, ties, tsvm
from tests.reference.dc_reference import dc_merge as dc_oracle
from tests.reference.iso_reference import iso_cts as iso_oracle
from tests.reference.tsvm_reference import compute_and_sum_svd_mem_reduction as tsvm_oracle
from tests.reference.ties_reference import ties_reference
from tests.reference.gse_reference import compute_component_selection_tsv_merge as gse_oracle


def context(**kwargs):
    return MergeContext(RuntimeSpec(**kwargs))


def problem(shape=(8, 8), count=3, kind="random"):
    generator = torch.Generator().manual_seed(791)
    base = {"matrix": torch.randn(shape, generator=generator),
            "bias": torch.randn(shape[0], generator=generator),
            "text_projection": torch.randn((2, 3), generator=generator),
            "counter": torch.tensor(4)}
    tasks = {}
    for i in range(count):
        update = torch.randn(shape, generator=generator)
        if kind == "one_zero" and i == 0:
            update.zero_()
        if kind == "all_zero":
            update.zero_()
        if kind == "low_rank":
            update = torch.randn((shape[0], 1), generator=generator) @ torch.randn((1, shape[1]), generator=generator)
        if kind == "tiny":
            update *= 1e-7
        tasks[str(i)] = {"matrix": base["matrix"] + update,
                         "bias": base["bias"] + torch.randn(shape[0], generator=generator),
                         "text_projection": base["text_projection"] + torch.randn((2, 3), generator=generator),
                         "counter": base["counter"].clone()}
    return make_inputs(base, tasks)


def oracle_weights(inputs, function, alpha=1.0, **kwargs):
    vectors = [SimpleNamespace(vector={name: inputs.delta(task, name)
                                      for name in inputs.parameter_order
                                      if inputs.base.tensor(name).is_floating_point()})
               for task in inputs.task_order]
    config = SimpleNamespace(DATASETS=inputs.task_order, device="cpu", method=SimpleNamespace(**kwargs))
    with redirect_stdout(io.StringIO()):
        update = function(vectors, config)
    return {name: inputs.base.tensor(name) + alpha * update[name]
            if name in update else inputs.base.tensor(name).clone()
            for name in inputs.parameter_order}


class ReferenceMethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assertStateEqual(self, actual, expected, *, exact=True):
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            if exact:
                self.assertTrue(torch.equal(actual[name], expected[name]), name)
            else:
                torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)

    def test_tsvm_frozen_reference_shapes_and_degeneracies(self):
        for shape in ((8, 8), (11, 6), (6, 11), (2, 2)):
            for kind in ("random", "one_zero", "all_zero", "low_rank", "tiny"):
                with self.subTest(shape=shape, kind=kind):
                    inputs = problem(shape=shape, kind=kind)
                    expected = oracle_weights(inputs, tsvm_oracle, alpha=.7)
                    actual = tsvm.merge(inputs, tsvm.TsvmParams(alpha=.7), context()).weights
                    self.assertStateEqual(actual, expected)

    def test_heavy_trace_time_is_separate_and_cached_across_alpha(self):
        inputs = problem()
        runtime = context(trace="full")
        tsvm.merge(inputs, tsvm.TsvmParams(), runtime)
        trace_seconds = runtime.timings["trace_seconds"]
        self.assertGreater(trace_seconds, 0)
        self.assertGreater(runtime.timings["merge_total_seconds"], trace_seconds)
        tsvm.merge(inputs, tsvm.TsvmParams(alpha=.5), runtime)
        self.assertEqual(trace_seconds, runtime.timings["trace_seconds"])

    def test_iso_cts_frozen_official_reference(self):
        for shape in ((8, 8), (11, 6), (6, 11)):
            for fraction in (0., .8, 1.):
                inputs = problem(shape=shape)
                with self.subTest(shape=shape, fraction=fraction):
                    r = min(shape)
                    if r - round((r - int(r * fraction)) / 3) * 3 < 0:
                        with self.assertRaises(ValueError):
                            iso_cts.merge(inputs, iso_cts.IsoCtsParams(.8, fraction), context())
                        continue
                    expected = oracle_weights(inputs, iso_oracle, alpha=.8, common_space_fraction=fraction)
                    actual = iso_cts.merge(inputs, iso_cts.IsoCtsParams(.8, fraction), context()).weights
                    self.assertStateEqual(actual, expected)

    def test_dc_merge_frozen_official_reference(self):
        for shape in ((8, 8), (11, 6), (6, 11), (2, 2)):
            for kind in ("random", "one_zero", "all_zero"):
                with self.subTest(shape=shape, kind=kind):
                    inputs = problem(shape=shape, kind=kind)
                    self.assertStateEqual(dc_merge.merge(inputs, dc_merge.DcMergeParams(.8), context()).weights,
                                          oracle_weights(inputs, dc_oracle, alpha=.8))

    def test_global_ties_matches_reference_for_all_block_sizes(self):
        inputs = problem()
        names = [name for name in sorted(inputs.parameter_order) if inputs.base.tensor(name).is_floating_point()]
        rows = torch.stack([torch.cat([inputs.delta(task, name).flatten() for name in names]) for task in inputs.task_order])
        for fraction in (.01, .2, 1.):
            expected = ties_reference(rows, fraction)
            for block_size in (1, 3, 262144):
                with self.subTest(fraction=fraction, block_size=block_size):
                    actual = ties.merge(inputs, ties.TiesParams(.75, fraction), context(block_size=block_size)).weights
                    offset = 0
                    for name in names:
                        base = inputs.base.tensor(name)
                        delta = expected[offset:offset + base.numel()].view_as(base)
                        self.assertTrue(torch.equal(actual[name], base + .75 * delta), name)
                        offset += base.numel()

    def test_global_ties_zero_sign_and_threshold_ties(self):
        base = {"a": torch.zeros(3), "b": torch.zeros(3)}
        task_a = {"a": torch.tensor([2., -2., 1.]), "b": torch.tensor([2., 0., -2.])}
        task_b = {"a": torch.tensor([-2., 2., 1.]), "b": torch.tensor([-2., 0., -2.])}
        inputs = make_inputs(base, {"a": task_a, "b": task_b})
        rows = torch.stack([torch.cat(list(task.values())) for task in (task_a, task_b)])
        expected = ties_reference(rows, .5)
        actual = ties.merge(inputs, ties.TiesParams(1., .5), context(block_size=2)).weights
        self.assertTrue(torch.equal(torch.cat(list(actual.values())), expected))

    def test_global_blocks_preserve_reference_reduction_tails_at_benchmark_task_counts(self):
        # A per-parameter or narrow-tail reduction changes float summation order
        # for T>=8, and can flip elected signs. Check real benchmark task counts.
        for count, size, dtype in itertools.product((8, 14, 20), (69, 1216, 1219), (torch.float32, torch.float64)):
            generator = torch.Generator().manual_seed(12)
            rows = torch.randn((count, size), generator=generator, dtype=dtype)
            cut = size // 3
            base = {"a": torch.zeros(cut, dtype=dtype), "b": torch.zeros(size - cut, dtype=dtype)}
            tasks = {str(i): {"a": rows[i, :cut], "b": rows[i, cut:]} for i in range(count)}
            inputs = make_inputs(base, tasks)
            expected = ties_reference(rows, .2)
            for block_size in (1, 17, 64, 128):
                with self.subTest(count=count, size=size, dtype=dtype, block=block_size):
                    actual = ties.merge(inputs, ties.TiesParams(), context(block_size=block_size)).weights
                    self.assertTrue(torch.equal(torch.cat([actual["a"], actual["b"]]), expected))

    def test_arithmetic_and_average_reference_sequence(self):
        inputs = problem()
        for fraction in (.2, 1.):
            names = [name for name in inputs.parameter_order if inputs.base.tensor(name).is_floating_point()]
            rows = torch.stack([torch.cat([inputs.delta(task, name).flatten() for name in names]) for task in inputs.task_order])
            if fraction != 1:
                threshold = rows.abs().kthvalue(rows.shape[1] - int(rows.shape[1] * fraction), dim=1, keepdim=True).values
                rows *= rows.abs() >= threshold
            expected_update = rows.sum(dim=0)
            for method, params, alpha in ((arithmetic.merge_average, arithmetic.AverageParams(fraction), 1 / 3),
                                         (arithmetic.merge_task_arithmetic, arithmetic.TaskArithmeticParams(.8, fraction), .8)):
                actual = method(inputs, params, context(block_size=7)).weights
                offset = 0
                for name in names:
                    base = inputs.base.tensor(name)
                    delta = expected_update[offset:offset + base.numel()].view_as(base)
                    self.assertTrue(torch.equal(actual[name], base + alpha * delta), name)
                    offset += base.numel()

    def test_zeroshot_is_exact_base(self):
        inputs = problem()
        self.assertStateEqual(arithmetic.merge_zeroshot(inputs, arithmetic.ZeroShotParams(), context()).weights,
                              {name: inputs.base.tensor(name) for name in inputs.parameter_order})

    def test_alpha_reuses_prepared_svd_update(self):
        inputs, ctx = problem(), context()
        with patch("src.methods.tsvm.svd", wraps=torch.linalg.svd) as mocked:
            tsvm.merge(inputs, tsvm.TsvmParams(.5), ctx)
            calls = mocked.call_count
            tsvm.merge(inputs, tsvm.TsvmParams(1.5), ctx)
            self.assertEqual(mocked.call_count, calls)
        self.assertEqual(set(ctx.cache), {"prepared_update"})
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in ctx.cache["prepared_update"][1].values()))

    def test_trace_levels_do_not_change_weights(self):
        inputs = problem()
        for module, params in ((tsvm, tsvm.TsvmParams()), (gse_tsv, gse_tsv.GseTsvParams())):
            results = [module.merge(inputs, params, context(trace=level)) for level in ("none", "summary", "full")]
            self.assertEqual(results[0].trace, {})
            self.assertIn("reconstruction", results[2].trace["layers"]["matrix"])
            for result in results[1:]:
                self.assertStateEqual(result.weights, results[0].weights)

    def test_trace_capture_level_change_does_not_reuse_incomplete_trace(self):
        inputs, ctx = problem(), context(trace="none")
        expected = tsvm.merge(inputs, tsvm.TsvmParams(), ctx).weights
        ctx.runtime = replace(ctx.runtime, trace="full")
        actual = tsvm.merge(inputs, tsvm.TsvmParams(), ctx)
        self.assertStateEqual(actual.weights, expected)
        self.assertIn("reconstruction", actual.trace["layers"]["matrix"])

    def test_global_methods_preserve_flatten_dtype_promotion(self):
        generator = torch.Generator().manual_seed(33)
        base = {"a": torch.randn(31, generator=generator), "b": torch.randn(38, generator=generator).double()}
        tasks = {str(i): {name: value + torch.randn(value.shape, generator=generator).to(value.dtype)
                         for name, value in base.items()} for i in range(8)}
        inputs = make_inputs(base, tasks)
        flat_base = torch.cat([base[name] for name in sorted(base)])
        rows = torch.stack([torch.cat([source[name] for name in sorted(base)]) - flat_base for source in tasks.values()])
        expected = ties_reference(rows, .2)
        actual = ties.merge(inputs, ties.TiesParams(), context(block_size=1)).weights
        offset = 0
        for name in sorted(base):
            delta = expected[offset:offset + base[name].numel()].to(base[name].dtype)
            self.assertTrue(torch.equal(actual[name], base[name] + delta))
            offset += base[name].numel()

    def test_inputs_are_not_modified_by_any_kernel(self):
        inputs = problem(count=2)
        snapshots = {(task, name): inputs.finetuned[task].tensor(name).clone()
                     for task in inputs.task_order for name in inputs.parameter_order}
        methods = ((arithmetic.merge_average, arithmetic.AverageParams()),
                   (arithmetic.merge_task_arithmetic, arithmetic.TaskArithmeticParams()),
                   (ties.merge, ties.TiesParams()), (tsvm.merge, tsvm.TsvmParams()),
                   (gse_tsv.merge, gse_tsv.GseTsvParams()), (iso_cts.merge, iso_cts.IsoCtsParams()),
                   (dc_merge.merge, dc_merge.DcMergeParams()), (sera.merge, sera.SeraParams()))
        for method, params in methods:
            method(inputs, params, context())
        consensus.merge(inputs, consensus.ConsensusParams(), context(), dict.fromkeys(inputs.task_order, .3))
        for (task, name), expected in snapshots.items():
            self.assertTrue(torch.equal(inputs.finetuned[task].tensor(name), expected))

    def test_nonfloating_buffers_are_copied(self):
        inputs = problem(count=2)
        result = tsvm.merge(inputs, tsvm.TsvmParams(), context())
        self.assertEqual(result.weights["counter"].dtype, torch.int64)
        self.assertEqual(result.weights["counter"].item(), 4)

    def test_global_raw_energy_is_not_task_normalized(self):
        spectra = [torch.tensor([10., 9., 8.]), torch.tensor([1., .5, .1])]
        self.assertEqual(gse_tsv.select_raw_energy_components(spectra, 1e-12), [3, 0])

    def test_gse_matches_frozen_workspace_reference(self):
        def reference(vectors, cfg):
            return gse_oracle(vectors, cfg, "global_raw_top_r", "GlobalRawEnergyTopR", False)
        for kind in ("random", "one_zero", "all_zero", "tiny"):
            inputs = problem(kind=kind)
            with self.subTest(kind=kind):
                expected = oracle_weights(inputs, reference, alpha=.9, eps=1e-12)
                actual = gse_tsv.merge(inputs, gse_tsv.GseTsvParams(.9), context()).weights
                self.assertStateEqual(actual, expected)

    def test_trace_levels_preserve_all_kernels_and_integer_buffers(self):
        inputs = problem(count=2)
        cases = ((arithmetic.merge_average, arithmetic.AverageParams()),
                 (ties.merge, ties.TiesParams()), (iso_cts.merge, iso_cts.IsoCtsParams()),
                 (dc_merge.merge, dc_merge.DcMergeParams()), (sera.merge, sera.SeraParams()))
        for method, params in cases:
            with self.subTest(method=method.__module__):
                expected = method(inputs, params, context(trace="none")).weights
                actual = method(inputs, params, context(trace="full")).weights
                self.assertStateEqual(actual, expected)

    def test_tsvm_supports_explicit_float64_reference(self):
        inputs = problem()
        base = {name: inputs.base.tensor(name).double() if inputs.base.tensor(name).is_floating_point() else inputs.base.tensor(name)
                for name in inputs.parameter_order}
        tasks = {task: {name: inputs.finetuned[task].tensor(name).double() if inputs.base.tensor(name).is_floating_point() else inputs.base.tensor(name)
                        for name in inputs.parameter_order} for task in inputs.task_order}
        double_inputs = make_inputs(base, tasks)
        self.assertStateEqual(tsvm.merge(double_inputs, tsvm.TsvmParams(.8), context(compute_dtype="float64")).weights,
                              oracle_weights(double_inputs, tsvm_oracle, .8))

    def test_prepared_cache_does_not_mix_task_order_or_method(self):
        inputs, ctx = problem(), context()
        tsvm.merge(inputs, tsvm.TsvmParams(), ctx)
        reversed_inputs = make_inputs(inputs.base, inputs.finetuned, tuple(reversed(inputs.task_order)))
        actual = tsvm.merge(reversed_inputs, tsvm.TsvmParams(), ctx).weights
        self.assertStateEqual(actual, oracle_weights(reversed_inputs, tsvm_oracle))
        actual = dc_merge.merge(inputs, dc_merge.DcMergeParams(), ctx).weights
        self.assertStateEqual(actual, oracle_weights(inputs, dc_oracle))

    def test_consensus_masks_match_original_arithmetic_order(self):
        inputs = problem(count=2)
        ctx = context()
        selected = {"0": .2, "1": .6}
        actual = consensus.merge(inputs, consensus.ConsensusParams(.7, 2), ctx, selected).weights
        for name in inputs.parameter_order:
            base = inputs.base.tensor(name)
            if not base.is_floating_point():
                continue
            rows = torch.stack([inputs.delta(task, name) for task in inputs.task_order])
            summed = rows.sum(dim=0)
            count = torch.zeros_like(base)
            for task in inputs.task_order:
                ft = inputs.finetuned[task].tensor(name)
                count += (base - ft).abs() > (base + summed - ft).abs() * selected[task]
            self.assertTrue(torch.equal(actual[name], base + .7 * summed * (count >= 2)))

    def test_consensus_reconstruction_has_task_specific_mask(self):
        inputs = problem(count=2)
        actual = consensus.reconstruction(inputs, "0", .3, context()).weights
        for name in inputs.parameter_order:
            base = inputs.base.tensor(name)
            if not base.is_floating_point():
                continue
            summed = torch.stack([inputs.delta(task, name) for task in inputs.task_order]).sum(dim=0)
            ft = inputs.finetuned["0"].tensor(name)
            expected = base + summed * ((base - ft).abs() > (base + summed - ft).abs() * .3)
            self.assertTrue(torch.equal(actual[name], expected))

    def test_consensus_cannot_silently_skip_fitting(self):
        with self.assertRaises(ValueError):
            consensus.merge(problem(), consensus.ConsensusParams(), context())

    def test_energy_rank_endpoint_is_bounded(self):
        for seed in range(100):
            s = torch.rand(64, generator=torch.Generator().manual_seed(seed)).sort(descending=True).values
            self.assertLessEqual(rank_for_energy(s, 1.), 64)
        self.assertEqual(rank_for_energy(torch.zeros(5), .95), 0)

    def test_sera_integer_projection_is_global_l2_minimum(self):
        demands, budget, max_rank = [1, 3, 4], 6, 4
        actual = sera.project_integer_budget(demands, budget, max_rank)
        target = [budget * x / sum(demands) for x in demands]
        objective = lambda values: sum((a - b) ** 2 for a, b in zip(values, target))
        minimum = min(objective(values) for values in itertools.product(range(1, max_rank + 1), repeat=3) if sum(values) == budget)
        self.assertAlmostEqual(objective(actual), minimum)
        self.assertEqual(sera.project_integer_budget([1, 1], 3, 3), [2, 1])

    def test_sera_allocates_different_capacity_at_fixed_budget(self):
        spectra = [torch.tensor([5., 0., 0., 0.]), torch.ones(4)]
        self.assertEqual(sera.allocate_ranks(spectra, .95), [1, 3])
        self.assertEqual(sum(sera.allocate_ranks(spectra, .95)), 4)

    def test_sera_zero_updates_remain_zero_and_infeasible_budget_is_explicit(self):
        inputs = make_inputs({"matrix": torch.zeros(4, 4)}, {"a": {"matrix": torch.zeros(4, 4)}, "b": {"matrix": torch.zeros(4, 4)}})
        actual = sera.merge(inputs, sera.SeraParams(), context()).weights
        self.assertTrue(torch.equal(actual["matrix"], torch.zeros(4, 4)))
        with self.assertRaises(ValueError):
            sera.allocate_ranks([torch.ones(2)] * 3, .95)

    def test_invalid_parameters_are_rejected(self):
        for factory in (lambda: ties.TiesParams(retention_fraction=0),
                        lambda: tsvm.TsvmParams(float("nan")),
                        lambda: sera.SeraParams(energy_threshold=1),
                        lambda: gse_tsv.GseTsvParams(eps=-1),
                        lambda: consensus.ConsensusParams(min_tasks=-1)):
            with self.assertRaises(ValueError):
                factory()


if __name__ == "__main__":
    unittest.main()
