"""Run one experiment, or a serial Hydra matrix with -m."""
import hydra
from omegaconf import DictConfig

from src.experiments.config import resolve_config


@hydra.main(config_path="config", config_name="experiment", version_base="1.3")
def main(config: DictConfig):
    from src.experiments.runner import run_experiment
    resume_from = config.get("resume_from")
    result = run_experiment(resolve_config(config), run_dir=resume_from,
                            resume=resume_from is not None)
    print(f"Saved experiment: {result['run_dir']}")
    print(f"Test macro accuracy: {result['test']['avg_top1']:.6f}")
    return result


if __name__ == "__main__":
    main()
