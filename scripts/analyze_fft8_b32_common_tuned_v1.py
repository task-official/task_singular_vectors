"""Analyze the saved TSV-M/GSE-TSV runs, without merging or evaluating again."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.analysis import load_results
from src.experiments.rank_analysis import analyze_rank_accuracy, save_rank_analysis

EXPERIMENT_ID = "fft8_b32_common_tuned_v1"
REPRESENTATIVE_LAYERS = tuple(f"model.visual.transformer.resblocks.{block}.attn.in_proj_weight"
                              for block in (0, 5, 11))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", help="Explicit source result.json files or run directories")
    parser.add_argument("--output", default=f"results/reports/{EXPERIMENT_ID}/tsvm_gse_rank_accuracy",
                        help="New report directory; never overwrite")
    parser.add_argument("--representative-layers", nargs="+", default=REPRESENTATIVE_LAYERS)
    args = parser.parse_args()
    paths = args.runs or sorted(Path(f"results/experiments/{EXPERIMENT_ID}").glob("*/result.json"))
    analysis = analyze_rank_accuracy(load_results(paths))
    if analysis.sources["tsvm"]["config"]["model"] != "ViT-B-32" or len(analysis.tasks) != 8:
        parser.error("This experiment entry expects ViT-B/32 and eight tasks")
    destination = save_rank_analysis(analysis, args.output, args.representative_layers)
    print(f"Saved rank/accuracy analysis: {destination}")
    print(f"Observed matrices: {len(analysis.layers)}; tasks: {len(analysis.tasks)}")


if __name__ == "__main__":
    main()
