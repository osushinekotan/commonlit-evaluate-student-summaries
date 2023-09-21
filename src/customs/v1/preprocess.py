from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.utils.logger import Logger

logger = Logger(__name__)


def run_preprocess(prompts_df: pd.DataFrame, summaries_df: pd.DataFrame):
    output_df = pd.merge(summaries_df, prompts_df, how="left", on="prompt_id")
    return output_df


@hydra.main(version_base=None, config_path="/workspace/configs/", config_name="config")
def run(cfg: DictConfig) -> None:
    filepath = Path(cfg.paths.misc_dir) / "train.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.is_file():
        prompts_train_df = pd.read_csv(Path(cfg.paths.input_dir) / "prompts_train.csv")
        summaries_train_df = pd.read_csv(Path(cfg.paths.input_dir) / "summaries_train.csv")
        train_df = run_preprocess(prompts_df=prompts_train_df, summaries_df=summaries_train_df)
        train_df.to_csv(filepath, index=None)
        logger.info(f"save : {filepath}")
    else:
        logger.info("train.csv already exists.")


if __name__ == "__main__":
    with logger.profile():
        run()
