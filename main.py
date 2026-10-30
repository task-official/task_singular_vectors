"""Translate historical settings into the single experiment runner."""
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config/legacy", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig):
    from src.compat.cli import translate_legacy_config
    from src.experiments.runner import run_experiment
    result = run_experiment(translate_legacy_config(cfg))
    print(f"Saved experiment: {result['run_dir']}")
    return result


if __name__ == "__main__":
    my_app()
