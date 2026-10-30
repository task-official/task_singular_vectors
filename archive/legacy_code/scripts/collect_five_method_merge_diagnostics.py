import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.eval.aggregation import get_all_checkpoints
from src.models.task_vectors import NonLinearTaskVector
from src.utils.TSVM_utils import compute_component_selection_tsv_merge
from src.utils.variables_and_paths import ALL_DATASETS


METHODS = {
    "TSVM": "tsvm",
    "AdaptiveTSV": "adaptive",
    "GlobalEnergyTopR": "global_top_r",
    "PerTask95FullConcat": "per_task_95",
    "FullComponentFullConcat": "full",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-location", default="models/checkpoints")
    parser.add_argument(
        "--output",
        default="results/five_method_comparison/ViT-B-32_8tasks_merge_diagnostics.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(args):
    config_dir = str((REPOSITORY_ROOT / "config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        config = compose(config_name="config", overrides=["method=AdaptiveTSV"])
    with open_dict(config):
        config.model = args.model
        config.num_tasks = args.num_tasks
        config.device = args.device
        config.model_location = args.model_location
        config.DATASETS = ALL_DATASETS[: args.num_tasks]
        config.DATASETS_VAL = [dataset + "Val" for dataset in config.DATASETS]
    return config


def measure_method(task_vectors, config, method_name, selection_mode):
    use_cuda = str(config.device).startswith("cuda")
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(config.device)
        torch.cuda.synchronize(config.device)
    started = time.perf_counter()
    merged, diagnostics = compute_component_selection_tsv_merge(
        task_vectors,
        config,
        selection_mode=selection_mode,
        method_name=method_name,
        return_diagnostics=True,
    )
    if use_cuda:
        torch.cuda.synchronize(config.device)
    diagnostics["performance"] = {
        "merge_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(config.device)) if use_cuda else None
        ),
    }
    del merged
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return diagnostics


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output}. Choose another --output path."
        )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    config = load_config(args)
    finetuned, pretrained = get_all_checkpoints(config)
    task_vectors = [
        NonLinearTaskVector(config.model, pretrained, checkpoint)
        for checkpoint in finetuned
    ]
    payload = {
        "model": config.model,
        "num_tasks": config.num_tasks,
        "datasets": list(config.DATASETS),
        "energy_threshold": float(config.method.energy_threshold),
        "eps": float(config.method.eps),
        "methods": {},
    }
    for method_name, selection_mode in METHODS.items():
        payload["methods"][method_name] = measure_method(
            task_vectors, config, method_name, selection_mode
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved diagnostics to {output}")


if __name__ == "__main__":
    main()
