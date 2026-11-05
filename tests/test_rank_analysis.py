"""Check energy aggregation, task-level inference and read-only reporting."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.experiments.analysis import load_results
from src.experiments.rank_analysis import analyze_rank_accuracy, correlation_rows, save_rank_analysis


FIRST = "model.visual.transformer.resblocks.0.attn.in_proj_weight"
SECOND = "model.visual.transformer.resblocks.10.attn.in_proj_weight"


def fixture(method):
    layers = {}
    for name, spectra in ((FIRST, {"A": [4., 3., 2., 1.], "B": [3., 2., 1., 0.]}),
                           (SECOND, {"A": [2., 1., 1., 0.], "B": [8., 4., 2., 1.]}),
                           ("model.positional_embedding", {"A": [0., 0., 0.], "B": [0., 0., 0.]})):
        if "visual" not in name:
            ranks = {"A": 1, "B": 1} if method == "tsvm" else {"A": 0, "B": 0}
        elif name == FIRST and method == "gse_tsv":
            ranks = {"A": 4, "B": 0}
        else:
            ranks = {"A": 2, "B": 2}
        layers[name] = {"r": len(spectra["A"]), "ranks": ranks, "singular_values": spectra,
                        "total_rank": sum(ranks.values()),
                        "total_energy": {task: sum(value*value for value in s) for task, s in spectra.items()},
                        "selected_energy_sum": {task: sum(value*value for value in s[:ranks[task]]) for task, s in spectra.items()}}
    values = {"A": .6, "B": .8} if method == "tsvm" else {"A": .7, "B": .75}
    per_task = {task: {"correct": round(value*100), "total": 100, "top1": value,
                       "normalized_top1": value/.9} for task, value in values.items()}
    return {"schema_version": 1, "status": "complete", "config": {
        "model": "synthetic", "benchmark": "B", "tasks": [{"id": "B"}, {"id": "A"}]},
        "input_identity": {"weights": "same"}, "validation_identity": {}, "test_identity": {},
        "method": {"id": method, "implementation": "fixture"}, "selected_params": {"alpha": 1.},
        "weights_fingerprint": method, "test": {"per_task": per_task,
        "avg_top1": sum(values.values())/2, "avg_normalized_top1": sum(values.values())/2/.9},
        "diagnostics": {"status": "collected", "observed": {"layers": layers}}}


class RankAnalysisTests(unittest.TestCase):
    def test_saved_ranks_r90_and_energy_weighting_are_not_averaged_ratios(self):
        records = [fixture("tsvm"), fixture("gse_tsv")]
        original = copy.deepcopy(records)
        analysis = analyze_rank_accuracy(records)
        self.assertEqual(analysis.tasks, ("B", "A"))
        self.assertEqual(len(analysis.layer_task_rows), 6)
        first_a = next(row for row in analysis.layer_task_rows if row["layer"] == FIRST and row["task"] == "A")
        self.assertEqual((first_a["r90"], first_a["r90_over_q"]), (3, .75))
        self.assertEqual(first_a["rank_change_fraction"], 1.)
        self.assertAlmostEqual(first_a["retained_energy_fraction_tsvm"], 25/30)
        row = next(row for row in analysis.task_rows if row["scope"] == "image_encoder" and row["task"] == "A")
        self.assertEqual((row["k_tsvm"], row["k_gse_tsv"]), (4, 6))
        self.assertEqual(row["rank_change_fraction"], .5)
        self.assertAlmostEqual(row["retained_energy_fraction_tsvm"], 30/36)
        self.assertAlmostEqual(row["mean_layer_retained_energy_fraction_tsvm"], (25/30+5/6)/2)
        self.assertAlmostEqual(row["retained_energy_fraction_gse_tsv"], 35/36)
        self.assertNotAlmostEqual(row["retained_energy_fraction_gse_tsv"], row["mean_layer_retained_energy_fraction_gse_tsv"])
        self.assertAlmostEqual(row["delta_accuracy_pp"], 10.)
        all_row = next(row for row in analysis.task_rows if row["scope"] == "all_recorded_matrices" and row["task"] == "A")
        self.assertEqual((all_row["k_tsvm"], all_row["k_gse_tsv"]), (5, 6))
        self.assertEqual(all_row["retained_energy_fraction_tsvm"], row["retained_energy_fraction_tsvm"])
        self.assertEqual(records, original)

    def test_zero_energy_is_undefined_and_zero_kept_rank_is_a_valid_observation(self):
        analysis = analyze_rank_accuracy([fixture("tsvm"), fixture("gse_tsv")])
        zero = next(row for row in analysis.layer_task_rows if row["layer"] == "model.positional_embedding")
        for key in ("r90", "r90_over_q", "retained_energy_fraction_tsvm", "delta_retained_energy_fraction"):
            self.assertIsNone(zero[key])
        self.assertEqual(zero["rank_change_fraction"], -1.)
        b = next(row for row in analysis.layer_task_rows if row["layer"] == FIRST and row["task"] == "B")
        self.assertEqual(b["k_gse_tsv"], 0)
        self.assertEqual(b["retained_energy_fraction_gse_tsv"], 0.)
        excluded = [row for row in analysis.layer_correlations if row["group"] == "model.positional_embedding"]
        self.assertTrue(all(row["pearson_r"] is None for row in excluded))

    def test_correlations_use_tasks_not_task_layer_rows_and_handle_ties(self):
        analysis = analyze_rank_accuracy([fixture("tsvm"), fixture("gse_tsv")])
        self.assertTrue(all(row["n_tasks"] == 2 for row in analysis.correlations))
        row = next(row for row in analysis.correlations if row["group"] == "image_encoder" and row["predictor"] == "rank_change_fraction" and row["response"] == "delta_accuracy_pp")
        self.assertAlmostEqual(row["pearson_r"], 1.)
        tied = [{"x": 1, "delta_accuracy_pp": n, "delta_normalized_accuracy_pp": n} for n in range(4)]
        self.assertTrue(all(row["status"] == "constant_variable" and row["pearson_r"] is None
                            for row in correlation_rows(tied, "test", ["x"])))

    def test_incompatible_missing_duplicate_or_inconsistent_observations_are_rejected(self):
        first, second = fixture("tsvm"), fixture("gse_tsv")
        with self.assertRaisesRegex(ValueError, "Multiple"):
            analyze_rank_accuracy([first, second, second])
        with self.assertRaises(ValueError):
            analyze_rank_accuracy([first])
        for change in (lambda item: item.update(input_identity={"different": "weights"}),
                       lambda item: item["diagnostics"].update(status="not_collected"),
                       lambda item: item["diagnostics"]["observed"]["layers"][FIRST]["total_energy"].update(A=100),
                       lambda item: item["diagnostics"]["observed"]["layers"][FIRST]["singular_values"]["A"].__setitem__(0, 5)):
            modified = copy.deepcopy(second)
            change(modified)
            with self.assertRaises(ValueError):
                analyze_rank_accuracy([first, modified])

    def test_export_and_plots_leave_sources_unchanged_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for method in ("tsvm", "gse_tsv"):
                path = root / f"{method}.json"
                path.write_text(json.dumps(fixture(method)))
                paths.append(path)
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            analysis = analyze_rank_accuracy(load_results(paths))
            output = root / "analysis"
            with patch("src.experiments.rank_plotting._save", return_value=[]):
                save_rank_analysis(analysis, output, [FIRST])
            self.assertEqual(len(list(output.glob('*.csv'))), 6)
            with (output / "layer_task_metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            zero = next(row for row in rows if row["layer"] == "model.positional_embedding")
            self.assertEqual(zero["r90"], "")
            with (output / "spectral_curves.csv").open() as handle:
                curves = list(csv.DictReader(handle))
            self.assertEqual(len(curves), 8)
            self.assertAlmostEqual(float(curves[-1]["cumulative_energy_fraction"]), 1.)
            with self.assertRaises(FileExistsError):
                save_rank_analysis(analysis, output, [FIRST])
            for path in paths:
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before[path])


if __name__ == "__main__":
    unittest.main()
