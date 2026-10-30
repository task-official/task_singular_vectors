"""Convert trusted local checkpoints before an experiment; never download."""
import json

import hydra
from omegaconf import DictConfig

from src.experiments.config import resolve_config
from src.experiments.inputs import prepare_inputs


@hydra.main(config_path="config", config_name="experiment", version_base="1.3")
def main(config: DictConfig):
    print(json.dumps(prepare_inputs(resolve_config(config)), indent=2))


if __name__ == "__main__":
    main()
