import subprocess
from pathlib import Path

from kaggle import KaggleApi

from src.utils.logger import Logger

logger = Logger(__name__)


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
