"""Compare completed records after checking their measured evaluation identities."""
import argparse
from pathlib import Path

from src.experiments.analysis import export_csv, load_results, make_comparison_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="result.json files or run directories")
    parser.add_argument("--output", required=True, help="New CSV path (never overwrite)")
    parser.add_argument("--allow-mixed", action="store_true", help="Include explicitly labelled incompatible conditions")
    parser.add_argument("--plots", metavar="DIRECTORY", help="Create figures in a new directory")
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
    rows = make_comparison_table(records, allow_mixed=args.allow_mixed)
    for destination in (args.output, args.plots):
        if destination and Path(destination).exists():
            raise FileExistsError(destination)
    export_csv(rows, args.output)
    if args.plots:
        from src.experiments.plotting import plot_comparison
        plot_comparison(records, rows, args.plots)
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
