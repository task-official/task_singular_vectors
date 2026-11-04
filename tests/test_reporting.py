"""Verify reported metrics, percentage tables and plots without rerunning evaluation."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.experiments.analysis import (
    comparison_labels, export_task_tables, make_comparison_table, make_task_tables, select_methods,
)
from src.experiments.plotting import plot_comparison


ROOT = Path(__file__).resolve().parents[1]


def record(method="tsvm", offset=0):
    # JSON key order deliberately differs from the declared benchmark order.
    per_task = {
        "TaskA": {"top1": .456789 + offset, "normalized_top1": .95 + offset},
        "TaskB": {"top1": .876543 + offset, "normalized_top1": 1.25 + offset},
    }
    return {
        "schema_version": 1, "status": "complete",
        "config": {"model": "synthetic", "benchmark": "two_tasks", "protocol": "common_fixed",
                   "tasks": [{"id": "TaskB"}, {"id": "TaskA"}]},
        "input_identity": {"weights": "same"}, "validation_identity": {}, "test_identity": {},
        "method": {"id": method, "implementation": "fixture"},
        "selected_params": {"alpha": 1.}, "weights_fingerprint": "fixture",
        "test": {"per_task": per_task, "avg_top1": .666666 + offset,
                 "avg_normalized_top1": 1.1 + offset},
    }


class ReportingTests(unittest.TestCase):
    def test_method_selection_preserves_order_and_inputs_and_rejects_missing_ids(self):
        records = [record(), record("iso_cts"), record("sera")]
        original = copy.deepcopy(records)
        selected = select_methods(records, ["sera", "tsvm"])
        self.assertEqual([item["method"]["id"] for item in selected], ["sera", "tsvm"])
        self.assertEqual(records, original)
        for names in ([], ["tsvm", "tsvm"], ["absent"]):
            with self.assertRaises(ValueError):
                select_methods(records, names)
        future = record("future_method")
        self.assertEqual(comparison_labels([future]), ["future_method"])
        self.assertEqual(len(set(comparison_labels([records[0], records[0]]))), 2)

    def test_tables_use_benchmark_order_percentages_and_unrounded_macro_means(self):
        records = [record(), record("iso_cts", .01)]
        tables = make_task_tables(records, make_comparison_table(records))
        self.assertEqual([table["metric"] for table in tables], ["top1", "normalized_top1"])
        absolute, normalized = tables
        row = absolute["rows"][0]
        self.assertEqual(list(row), ["Method", "TaskB", "TaskA", "Mean"])
        self.assertEqual(row["Method"], "TSV-M")
        self.assertAlmostEqual(row["TaskB"], 87.6543)
        self.assertAlmostEqual(row["Mean"], 66.6666)
        self.assertEqual(normalized["rows"][0]["TaskB"], 125.)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tables"
            paths = export_task_tables(tables, output)
            self.assertEqual(len(paths), 4)
            csv_path = next(path for path in paths if path.name.endswith("_accuracy_percent.csv")
                            and "normalized" not in path.name)
            with csv_path.open() as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(saved[0], {"Method": "TSV-M", "TaskB": "87.65", "TaskA": "45.68", "Mean": "66.67"})
            markdown = csv_path.with_suffix(".md").read_text()
            self.assertIn("| Method | TaskB | TaskA | Mean |", markdown)
            self.assertIn("| TSV-M | 87.65 | 45.68 | 66.67 |", markdown)
            with self.assertRaises(FileExistsError):
                export_task_tables(tables, output)
        malformed = copy.deepcopy(records)
        del malformed[0]["test"]["per_task"]["TaskB"]
        with self.assertRaisesRegex(ValueError, "task set"):
            make_task_tables(malformed, make_comparison_table(malformed))

    def test_plots_label_actual_metrics_show_values_grid_100_and_external_legend(self):
        records = [record(), record("iso_cts", .01)]
        original = copy.deepcopy(records)
        captured = {}

        def inspect(figure, output, stem):
            axis = figure.axes[0]
            figure.canvas.draw()
            captured[stem] = {
                "ylabel": axis.get_ylabel(),
                "texts": [text.get_text() for text in axis.texts],
                "ticks": axis.get_yticks(), "limits": axis.get_ylim(),
                "grid": any(line.get_visible() for line in axis.get_ygridlines()),
                "below": axis.get_axisbelow(),
                "xlabels": [label.get_text() for label in axis.get_xticklabels()],
                "lines": [list(line.get_ydata()) for line in axis.lines],
                "bar_heights": [bar.get_height() for bar in axis.patches],
            }
            legend = axis.get_legend()
            if legend is not None:
                renderer = figure.canvas.get_renderer()
                self.assertGreater(legend.get_window_extent(renderer).x0, axis.get_window_extent(renderer).x1)
            return []

        with tempfile.TemporaryDirectory() as directory, patch("src.experiments.plotting._save", side_effect=inspect):
            plot_comparison(records, make_comparison_table(records), Path(directory) / "figures", task_metric="both")
        self.assertEqual(len(captured), 4)
        means = next(value for key, value in captured.items() if key.endswith("_avg_top1"))
        norm_means = next(value for key, value in captured.items() if key.endswith("_avg_normalized_top1"))
        tasks = next(value for key, value in captured.items() if key.endswith("_task_accuracy"))
        norm_tasks = next(value for key, value in captured.items() if key.endswith("_task_normalized_accuracy"))
        self.assertEqual(means["texts"], ["66.67", "67.67"])
        self.assertEqual(norm_means["texts"], ["110.00", "111.00"])
        self.assertAlmostEqual(means["bar_heights"][0], 66.6666)
        self.assertEqual(norm_means["ylabel"], "Normalized Accuracy (%)")
        self.assertIn(100, norm_means["ticks"])
        self.assertGreater(norm_means["limits"][1], 111)
        self.assertEqual(tasks["ylabel"], "Accuracy (%)")
        self.assertEqual(norm_tasks["ylabel"], "Normalized Accuracy (%)")
        self.assertEqual(tasks["xlabels"], ["TaskB", "TaskA"])
        self.assertAlmostEqual(tasks["lines"][0][0], 87.6543)
        self.assertEqual(norm_tasks["lines"][0], [125., 95.])
        self.assertGreater(norm_tasks["limits"][1], 126)
        for figure in captured.values():
            self.assertTrue(figure["grid"])
            self.assertTrue(figure["below"])
        self.assertEqual(records, original)

    def test_unavailable_normalized_metric_is_not_relabelled_absolute_accuracy(self):
        item = record()
        item["test"]["per_task"]["TaskA"]["normalized_top1"] = None
        tables = make_task_tables([item], make_comparison_table([item]))
        self.assertEqual([table["metric"] for table in tables], ["top1"])
        with tempfile.TemporaryDirectory() as directory, patch("src.experiments.plotting._save", return_value=[]):
            with self.assertRaisesRegex(ValueError, "cannot relabel"):
                plot_comparison([item], make_comparison_table([item]), Path(directory) / "figures",
                                task_metric="normalized_accuracy")

    def test_actual_cli_exports_subset_plots_and_tables_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for method in ("tsvm", "iso_cts", "zeroshot"):
                path = root / f"{method}.json"
                path.write_text(json.dumps(record(method)))
                sources.append(path)
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
            destination = root / "report"
            command = [sys.executable, "-B", str(ROOT / "compare.py"), *map(str, sources),
                       "--methods", "iso_cts", "tsvm", "--task-metric", "both",
                       "--output", str(destination / "comparison.csv"),
                       "--plots", str(destination / "figures"), "--tables", str(destination / "tables")]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (destination / "comparison.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["method"] for row in rows], ["iso_cts", "tsvm"])
            self.assertEqual(float(rows[0]["avg_top1"]), .666666)
            self.assertEqual(len(list((destination / "figures").glob("*.png"))), 4)
            self.assertEqual(len(list((destination / "figures").glob("*.svg"))), 4)
            for extension in ("csv", "md", "png", "svg"):
                self.assertEqual(len(list((destination / "tables").glob(f"*.{extension}"))), 2)
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            for path, digest in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_cli_rejects_overlapping_destinations_before_writing(self):
        import compare
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report/comparison.csv"
            arguments = ["compare.py", "unused", "--output", str(output), "--tables", str(output.parent)]
            with patch("sys.argv", arguments), patch("compare.load_results", return_value=[record()]):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    compare.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
