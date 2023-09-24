import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from kaggle import KaggleApi
from omegaconf import DictConfig
from transformers import AutoConfig, AutoTokenizer

from src.utils.logger import HydraLogger

logger = HydraLogger(__name__)


def download_kaggle_competition_dataset(
    client: "KaggleApi",
    competition: str,
    out_dir: Path,
):
    zipfile_path = out_dir / f"{competition}.zip"

    if not zipfile_path.is_file():
        client.competition_download_files(
            competition=competition,
            path=out_dir,
            quiet=False,
        )
        subprocess.run(["unzip", "-q", zipfile_path, "-d", out_dir])
    else:
        logger.info("Dataset already exists.")


class Deploy:
    def __init__(
        self,
        cfg: DictConfig,
        client: "KaggleApi",
    ):
        self.cfg = cfg
        self.client = client

    def push_output(self) -> None:
        # model and predictions
        dataset_name = re.sub(r"[/_=]", "-", self.cfg.experiment_name)
        metadata = make_dataset_metadata(dataset_name=dataset_name)
        with tempfile.TemporaryDirectory() as tempdir:
            with open(Path(tempdir) / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
            self.client.dataset_create_new(
                folder=self.cfg.paths.output_dir,
                public=False,
                quiet=False,
            )

    def push_huguingface_model(self) -> None:
        # pretrained tokenizer and config
        model_name = self.cfg.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        dataset_name = re.sub(r"[/_]", "-", model_name)
        metadata = make_dataset_metadata(dataset_name=dataset_name)

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)
            tokenizer.save_pretrained(tempdir)
            with open(Path(tempdir) / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            self.client.dataset_create_new(
                folder=tempdir,
                public=True,
                quiet=False,
            )


def make_dataset_metadata(dataset_name: str) -> dict:
    dataset_metadata = {}
    dataset_metadata["id"] = f'{os.environ["KAGGLE_USERNAME"]}/{dataset_name}'
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = dataset_name
    return dataset_metadata
