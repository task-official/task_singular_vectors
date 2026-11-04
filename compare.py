"""Compare completed records after checking their measured evaluation identities."""
import argparse
from pathlib import Path

from src.experiments.analysis import (export_csv, export_task_tables, load_results,
                                      make_comparison_table, make_task_tables, select_methods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="result.json files or run directories")
    parser.add_argument("--output", required=True, help="New CSV path (never overwrite)")
    parser.add_argument("--allow-mixed", action="store_true", help="Include explicitly labelled incompatible conditions")
    parser.add_argument("--plots", metavar="DIRECTORY", help="Create figures in a new directory")
    parser.add_argument("--methods", nargs="+", help="Select method IDs in this display order")
    parser.add_argument("--task-metric", choices=("accuracy", "normalized_accuracy", "both"),
                        default="accuracy", help="Per-task plots; never relabel absolute accuracy as normalized")
    parser.add_argument("--tables", metavar="DIRECTORY", help="Create task-by-method percentage tables (CSV/Markdown/PNG/SVG)")
    parser.add_argument("--legacy", action="store_true", help="Explicit historical import; unknown conditions remain labelled")
    parser.add_argument("--legacy-labels", nargs="+", help="Explicit method labels, one per historical input")
    args = parser.parse_args()
    if args.legacy:
        from src.compat.results import legacy_view
        labels = args.legacy_labels or ["historical_unknown"] * len(args.paths)
        if len(labels) != len(args.paths):
            parser.error("--legacy-labels must match the input count")
        records = [legacy_view(path, label) for path, label in zip(args.paths, labels)]
    else:
        records = load_results(args.paths)
    if args.methods:
        records = select_methods(records, args.methods)
    rows = make_comparison_table(records, allow_mixed=args.allow_mixed)
    task_tables = make_task_tables(records, rows) if args.tables else None
    for destination in (args.output, args.plots, args.tables):
        if destination and Path(destination).exists():
            raise FileExistsError(destination)
    destinations = [Path(path).resolve() for path in (args.output, args.plots, args.tables) if path]
    for index, left in enumerate(destinations):
        for right in destinations[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("CSV, figure and table destinations must not overlap")
    export_csv(rows, args.output)
    if args.plots:
        from src.experiments.plotting import plot_comparison
        plot_comparison(records, rows, args.plots, task_metric=args.task_metric)
    if args.tables:
        from src.experiments.plotting import plot_task_tables
        export_task_tables(task_tables, args.tables)
        plot_task_tables(task_tables, args.tables)
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
