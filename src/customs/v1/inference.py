from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.customs.v1.preprocess import run_preprocess
from src.tools.torch.trainer import inference_fn
from src.utils.logger import HydraLogger

logger = HydraLogger(__name__)

# =================================
# inference (kaggle format)
# =================================


def inference_loop(cfg: DictConfig, test_df: pd.DataFrame, output_dir: Path) -> None:
    test_dataset = instantiate(cfg.dataset.test_dataset)(cfg=cfg, df=test_df)
    test_dataloader = instantiate(cfg.test_dataloader)(dataset=test_dataset)

    model = instantiate(cfg.model)(cfg=cfg, pretrained=False)
    state = torch.load(output_dir / "model.pth")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_output = inference_fn(
        model=model,
        dataloader=test_dataloader,
        device=device,
    )
    return test_output


def inference_fold(cfg: DictConfig, test_df: pd.DataFrame) -> np.ndarray:
    num_fold = cfg.num_fold
    valid_folds = cfg.valid_folds

    output_dir = Path(cfg.paths.artifact_dir)
    test_outputs = []

    for i_fold in range(num_fold):
        if i_fold not in valid_folds:
            continue
        i_test_output = inference_loop(cfg=cfg, test_df=test_df, output_dir=output_dir / f"fold_{i_fold}")
        test_outputs.append(i_test_output["outputs"])
    return np.mean(test_outputs, axis=0)


def preprocess(cfg: DictConfig) -> pd.DataFrame:
    prompts_test_df = pd.read_csv(Path(cfg.paths.input_dir) / "prompts_test.csv")
    summaries_test_df = pd.read_csv(Path(cfg.paths.input_dir) / "summaries_test.csv")
    test_df = run_preprocess(prompts_df=prompts_test_df, summaries_df=summaries_test_df)
    return test_df


def run(cfg: DictConfig) -> None:
    test_df = preprocess(cfg=cfg)
    test_output = inference_fold(cfg=cfg, test_df=test_df)

    dirpath = Path(cfg.paths.inference_dir)
    dirpath.mkdir(parents=True, exist_ok=True)
    joblib.dump(test_output, dirpath / "test_output.pkl")


@hydra.main(version_base=None, config_path="/workspace/configs/", config_name="config")
def main(cfg: DictConfig):
    run(cfg=cfg)


if __name__ == "__main__":
    main()
