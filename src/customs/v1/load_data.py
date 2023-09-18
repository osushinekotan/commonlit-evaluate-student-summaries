from pathlib import Path

import hydra
from kaggle import KaggleApi
from omegaconf import DictConfig

from src.utils.kaggle_utils import download_kaggle_competition_dataset
from src.utils.logger import Logger

logger = Logger(__name__)


@hydra.main(version_base=None, config_path="/workspace/configs/", config_name="config")
def run(cfg: DictConfig) -> None:
    client = KaggleApi()
    client.authenticate()

    download_kaggle_competition_dataset(
        client=client,
        competition=cfg.meta.competition,
        out_dir=Path(cfg.paths.input_dir),
    )


if __name__ == "__main__":
    run()
