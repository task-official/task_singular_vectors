"""Historical CLI compatibility; supported baselines share the new runner."""
import uuid

import hydra
from omegaconf import DictConfig, open_dict


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig):
    from src.compat.cli import ALIASES, translate_legacy_config
    name = cfg.method.name
    if name in ALIASES:
        from src.experiments.runner import run_experiment
        result = run_experiment(translate_legacy_config(cfg))
        print(f"Saved experiment: {result['run_dir']}")
        return result
    # Preserve historical task-specific and research modes in an isolated run.
    from src.compat.legacy_cli import run_legacy
    with open_dict(cfg):
        cfg.save_subfolder = f"experiments/legacy-{uuid.uuid4().hex}"
    print(f"Historical experimental adapter: {name}; results/{cfg.save_subfolder}")
    return run_legacy(cfg)


if __name__ == "__main__":
    my_app()
