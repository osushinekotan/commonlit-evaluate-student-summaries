from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.logger import HydraLogger

logger = HydraLogger(__name__)


def run_preprocess(prompts_df: pd.DataFrame, summaries_df: pd.DataFrame):
    output_df = pd.merge(summaries_df, prompts_df, how="left", on="prompt_id")
    return output_df


def assign_fold_idx(cfg: DictConfig, train_df: pd.DataFrame) -> pd.DataFrame:
    cv = instantiate(cfg.fold)
    train_df["fold"] = -1
    dummy_y = np.ones(len(train_df))
    for i_fold, (_, va_idx) in enumerate(cv.split(X=train_df, y=dummy_y, groups=train_df["prompt_id"])):
        train_df.loc[va_idx, "fold"] = i_fold
    return train_df


def run(cfg: DictConfig) -> None:
    filepath = Path(cfg.paths.misc_dir) / "train.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.is_file() and (not cfg.overwrite_preprocess):
        logger.info("train.csv already exists.")
        return

    # load raw data
    prompts_train_df = pd.read_csv(Path(cfg.paths.input_dir) / "prompts_train.csv")
    summaries_train_df = pd.read_csv(Path(cfg.paths.input_dir) / "summaries_train.csv")

    # preoprocess
    train_df = run_preprocess(prompts_df=prompts_train_df, summaries_df=summaries_train_df)
    train_df = assign_fold_idx(cfg=cfg, train_df=train_df)

    # save preprocessed data
    train_df.to_csv(filepath, index=None)
    logger.info(f"save : {filepath}")


@hydra.main(version_base=None, config_path="/workspace/configs/", config_name="config")
def main(cfg: DictConfig):
    with logger.profile():
        run(cfg=cfg)


if __name__ == "__main__":
    main()
