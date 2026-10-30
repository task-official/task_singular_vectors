"""Translate historical arguments; numerical semantics live in canonical kernels."""
from __future__ import annotations

from src.experiments.config import resolve_config
from src.utils.variables_and_paths import ALL_DATASETS

ALIASES = {"zeroshot": "zeroshot", "average": "weight_average",
           "sum": "task_arithmetic", "TSVM": "tsvm", "ties": "ties",
           "GlobalRawEnergyTopR": "gse_tsv", "consensus": "consensus_ta",
           "AdaptiveTSV": "adaptive_tsv", "SubspaceTSV": "subspace_tsv",
           "AdaptiveSubspaceTSV": "subspace_tsv", "GlobalEnergyTopR": "component_selection",
           "PerTask95FullConcat": "component_selection", "FullComponentFullConcat": "component_selection",
           "TSVM_2": "eigen_tsv", "dummy": "dummy"}


def translate_legacy_config(config):
    from omegaconf import OmegaConf
    import numpy as np
    cfg = OmegaConf.to_container(config, resolve=True)
    name = cfg["method"]["name"]
    if name not in ALIASES:
        raise ValueError(f"{name} is preserved in archive/legacy_code; task-specific experiments are outside this shared-model pipeline")
    if int(cfg.get("world_size", 1)) != 1:
        raise ValueError("The common runner supports one process/GPU; world_size must be 1")
    tasks = list(cfg["DATASETS"]) if cfg.get("DATASETS") else ALL_DATASETS[:int(cfg["num_tasks"])]
    k = float(cfg["method"].get("k", 100))
    retention = k / 100 if k >= 1 else k
    params = {}
    if name in ("average", "sum", "ties"):
        params["retention_fraction"] = retention
    if name == "zeroshot":
        candidates = [{}]
    elif name == "average":
        candidates = [{}]
    elif cfg.get("specify_lambda", "None") not in ("None", None):
        candidates = [{"alpha": float(cfg["specify_lambda"])}]
    else:
        maximum = 1.0 if name in ("sum", "ties", "consensus", "dummy") else 3.0
        candidates = [{"alpha": float(alpha)} for alpha in np.linspace(0, maximum, int(cfg["n_eval_points"]))[1:]]
    mode = "fixed" if len(candidates) == 1 else "grid"
    resolved = {
        "benchmark": {"name": f"legacy_{len(tasks)}tasks", "task_ids": tasks, "tasks": {}},
        "model": {"name": cfg["model"], "pretrained_tag": "openai"},
        "inputs": {"checkpoint_root": cfg["model_location"], "data_root": cfg["data_location"],
                   "base": "{model}/MNISTVal/nonlinear_zeroshot.pt",
                   "finetuned": "{model}/{task}Val/nonlinear_finetuned.pt",
                   "head": "{model}/head_{task}Val.pt", "source_status": "historical_local_unverified"},
        "method": {"id": ALIASES[name], "parameters": params},
        "protocol": {"name": "legacy_reference", "selection": {
            "mode": mode, "candidates": candidates, "seed": cfg.get("seed", 0),
            "objective": "first_task_top1" if name == "dummy" else "macro_normalized_top1",
            # Legacy search was unlimited; a huge explicit cap preserves its candidate list.
            "max_validation_images": None if mode == "fixed" else 2**63 - 1}},
        "runtime": {"device": cfg["device"], "batch_size": cfg["batch_size"], "num_workers": 0,
                    "cache_dir": cfg.get("cache_dir") if cfg.get("cache_dir") not in (None, "None", "") else "cache",
                    "output_dir": "results/" + str(cfg["save_subfolder"]) if cfg.get("save_subfolder") else "results/experiments",
                    "seed": cfg.get("seed", 0)},
    }
    if name == "consensus":
        if cfg["method"].get("use_ties", False) or retention != 1.0:
            raise ValueError("consensus_ta compatibility requires use_ties=false and k=100; other variants remain archived")
        params["min_tasks"] = int(cfg["method"].get("prun_thre_k", 2))
    if name in ("AdaptiveTSV", "SubspaceTSV", "AdaptiveSubspaceTSV", "GlobalEnergyTopR", "PerTask95FullConcat", "FullComponentFullConcat"):
        params.update(energy_threshold=cfg["method"].get("energy_threshold", .95), eps=cfg["method"].get("eps", 1e-12))
    if name in ("SubspaceTSV", "AdaptiveSubspaceTSV"):
        params["weighted"] = name == "AdaptiveSubspaceTSV"
    selections = {"GlobalEnergyTopR": "normalized_top_r", "PerTask95FullConcat": "per_task_energy", "FullComponentFullConcat": "full"}
    if name in selections:
        params["selection"] = selections[name]
    if name == "GlobalRawEnergyTopR":
        params["eps"] = cfg["method"].get("eps", 1e-12)
    return resolve_config(resolved)


def standalone_spec(args, benchmark="fft20"):
    """Resolve historical argparse entrypoints using the canonical YAML groups."""
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    models = {"ViT-B-32": "vit_b32", "ViT-B-16": "vit_b16", "ViT-L-14": "vit_l14"}
    if args.model not in models:
        raise ValueError(f"Unsupported model: {args.model}")
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "config"), version_base="1.3"):
        cfg = compose(config_name="experiment", overrides=[f"model={models[args.model]}", f"benchmark={benchmark}"])
    cfg.inputs.checkpoint_root = args.model_location
    cfg.inputs.data_root = args.data_location
    cfg.runtime.device = args.device
    cfg.runtime.batch_size = args.batch_size
    cfg.runtime.num_workers = args.num_workers
    cfg.runtime.seed = args.seed if args.seed is not None else 0
    cfg.runtime.cache_dir = args.cache_dir or "cache"
    cfg.runtime.output_dir = args.output_dir or ("results/" + args.save_subfolder if args.save_subfolder else "results/experiments")
    if args.eval_datasets:
        cfg.benchmark.task_ids = args.eval_datasets
        cfg.benchmark.name = "explicit_individual_tasks"
    return resolve_config(OmegaConf.to_container(cfg, resolve=True))


def single_task_main(benchmark="fft20", rank=False):
    from src.utils.args import parse_arguments
    from src.evaluation.single_task import run_individuals
    from src.experiments.rank_study import run_rank_study, DEFAULT_FRACTIONS
    args = parse_arguments()
    spec = standalone_spec(args, benchmark)
    if rank:
        results = run_rank_study(spec, args.rank_fractions or DEFAULT_FRACTIONS)
    else:
        results = run_individuals(spec, mode=args.finetuning_mode, reconstructed=not args.direct_ft)
    for result in results:
        print(f"Saved experiment: {result['run_dir']}")
    return results
