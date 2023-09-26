from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.logger import HydraLogger

logger = HydraLogger(__name__)


class SimpleEmsemble:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.seeds = [f"experiment={s}" for s in cfg.emsemble_seeds]
        self.metrics = instantiate(cfg.metrics)

    def _get_file_paths(self, filename: str) -> list[Path]:
        base_path = Path(self.cfg.paths.resource_dir) / "outputs"
        return [base_path / seed / "artifacts" / filename for seed in self.seeds]

    def _aggregate_predictions(self, predictions: list) -> np.ndarray:
        agg_method = self.cfg.agg_method
        if agg_method == "mean":
            return np.mean(predictions, axis=0)
        elif agg_method == "median":
            return np.median(predictions, axis=0)
        else:
            raise NotImplementedError(f"Aggregation method {agg_method} is not implemented")

    def evalate(self) -> None:
        oof_paths = self._get_file_paths("oof_output.pkl")
        oofs = [joblib.load(path) for path in oof_paths]
        oof_preds = [oof_df[["preds_content", "preds_wording"]].to_numpy() for oof_df in oofs]

        aggregated_preds = self._aggregate_predictions(oof_preds)

        oof_targets = oofs[0][["content", "wording"]].to_numpy()
        score = self.metrics(outputs=aggregated_preds, targets=oof_targets)
        logger.info(score)

    def inference(self) -> np.ndarray:
        preds_paths = self._get_file_paths("test_output.pkl")
        test_preds = [joblib.load(path) for path in preds_paths]

        return self._aggregate_predictions(test_preds)


def make_submission(cfg: DictConfig, test_preds: np.ndarray) -> None:
    submission_dir = Path(cfg.submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_df = pd.read_csv(Path(cfg.paths.input_dir) / "sample_submission.csv")
    submission_df[["content", "wording"]] = test_preds
    logger.debug(f"submission_df : \n{submission_df}")
    submission_df.to_csv(submission_dir / "submission.csv")


@hydra.main(version_base=None, config_path="/workspace/configs/", config_name="config")
def main(cfg: DictConfig):
    se = SimpleEmsemble(cfg=cfg)
    if cfg.evaluate:
        with logger.profile(target="evaluate"):
            se.evalate()

    with logger.profile(target="make_submission"):
        test_preds = se.inference()
        make_submission(cfg=cfg, test_preds=test_preds)


if __name__ == "__main__":
    main()
