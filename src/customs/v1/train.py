import gc
import os
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from src.tools.torch.trainer import train_fn, valid_fn
from src.utils.logger import HydraLogger
from src.utils.nn_utils import calc_steps

logger = HydraLogger(__name__)

# Log in to wandb using your API key.
wandb.login(key=os.environ["WANDB_KEY"])


def is_new_best_score(current_best_score: dict | float, new_score: dict) -> bool:
    """Determine if the new score is better than the current best score."""
    if current_best_score == -np.inf:
        return True
    for metric_name, metric_value in new_score.items():
        if current_best_score[metric_name] < metric_value:
            return True
    return False


def initialize_wandb(cfg: DictConfig, name: str) -> None:
    debug = "_debug" if cfg.debug else ""
    wandb.init(
        project=cfg.meta.competition,
        name=name,
        group=cfg.experiment_name + debug,
        job_type="train",
        anonymous=None,
        reinit=True,
    )


def collate_fn(data_list):
    batch = {key: torch.stack([item[key] for item in data_list]) for key in data_list[0].keys()}
    mask_len = int(batch["attention_mask"].sum(axis=1).max())
    for k, v in batch.items():
        if k != "targets":
            batch[k] = batch[k][:, :mask_len]
    return batch


def train_loop(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Training loop for a given experiment configuration, training and validation data."""

    # initialize wandb logger by one loop
    initialize_wandb(cfg=cfg, name=str(output_dir).split("/")[-1])  # NOTE : name is fold index

    # Instantiate datasets and dataloaders.
    train_dataset = instantiate(cfg.dataset.train_dataset)(cfg=cfg, df=train_df)
    valid_dataset = instantiate(cfg.dataset.train_dataset)(cfg=cfg, df=valid_df)
    train_loader = instantiate(cfg.train_dataloader)(dataset=train_dataset, collate_fn=collate_fn)
    valid_loader = instantiate(cfg.valid_dataloader)(dataset=valid_dataset, collate_fn=collate_fn)

    model = instantiate(cfg.model)(cfg=cfg)
    optimizer = instantiate(cfg.optimizer)(params=model.parameters())

    max_epochs = cfg.max_epochs
    if cfg.batch_scheduler:
        num_training_steps = calc_steps(
            iters_per_epoch=len(train_loader),
            max_epochs=max_epochs,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )  # Calcurate training step for batch scheduling
        scheduler = instantiate(cfg.scheduler)(optimizer=optimizer, num_training_steps=num_training_steps)
    else:
        scheduler = instantiate(cfg.scheduler)(optimizer=optimizer)

    metrics = instantiate(cfg.metrics)
    criterion = instantiate(cfg.criterion)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_steps = 0  # Track total steps across all epochs.
    best_score = -np.inf  # Initialize the best score to negative infinity. (upper is better)

    for epoch in range(max_epochs):
        train_output = train_fn(
            cfg=cfg,
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            total_step=total_steps,
            wandb_logger=wandb,
            device=device,
        )
        epoch_loss, epoch_step = train_output["loss"], train_output["step"]
        total_steps += epoch_step

        # Validate the model.
        validation_output = valid_fn(cfg=cfg, model=model, dataloader=valid_loader, criterion=criterion, device=device)

        # Evaluate the model's performance using the provided metrics.
        eval_results = metrics(outputs=validation_output["outputs"], targets=valid_df[cfg.target])

        # Log results.
        logs = {
            "epoch": epoch,
            "train_loss_epoch": epoch_loss,
            "valid_loss_epoch": validation_output["loss"],
            **eval_results,
        }
        logger.info(logs)
        wandb.log(logs)

        # Check if the current model has the best score and save it if it does.
        if is_new_best_score(current_best_score=best_score, new_score=eval_results):
            best_score = eval_results
            logger.info(f"epoch {epoch} - best score: {best_score} model 🌈")

            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / "model.pth")  # Save the model's state dict.
            joblib.dump(validation_output, output_dir / "output.pkl")  # Save the validation outputs.

    wandb.finish(quiet=True)
    torch.cuda.empty_cache()
    gc.collect()


def train_fold(cfg: DictConfig, train_df: pd.DataFrame) -> pd.DataFrame:
    num_fold = cfg.num_fold
    valid_folds = cfg.valid_folds
    overwrite_fold = cfg.overwrite_fold

    output_dir = Path(cfg.paths.artifact_dir)

    oof_outputs = []
    for i_fold in range(num_fold):
        if i_fold not in valid_folds:
            continue

        i_output_dir = output_dir / f"fold_{i_fold}"
        if (not i_output_dir.is_dir()) or (overwrite_fold):
            train_feature_df = train_df[train_df["fold"] != i_fold].reset_index(drop=True)
            valid_feature_df = train_df[train_df["fold"] == i_fold].reset_index(drop=True)

            with logger.profile(target=f"train : fold={i_fold}"):
                train_loop(
                    cfg=cfg,
                    train_df=train_feature_df,
                    valid_df=valid_feature_df,
                    output_dir=i_output_dir,
                )

        valid_out_df = pd.concat(
            [
                valid_feature_df[["student_id", "prompt_id", "content", "wording"]],
                pd.DataFrame(joblib.load(i_output_dir / "output.pkl")["outputs"], columns=cfg.target),
            ],
            axis=1,
        )
        oof_outputs.append(valid_out_df)
    return pd.concat(oof_outputs, axis=0).reset_index(drop=True)


def run(cfg: DictConfig) -> None:
    filepath = Path(cfg.paths.misc_dir) / "train.csv"
    train_df = pd.read_csv(filepath)

    if cfg.debug:
        train_df = train_df.sample(100, random_state=cfg.seed).reset_index(drop=True)

    logger.debug(f"train_df : {train_df.shape}")
    logger.debug(f"train_df : \n{train_df.head()}")

    oof_output: pd.DataFrame = train_fold(cfg=cfg, train_df=train_df)
    logger.debug(f"oof_output : \n{oof_output.head()}")
    joblib.dump(oof_output, Path(cfg.paths.artifact_dir) / "oof_output.pkl")


@hydra.main(version_base=None, config_path="/workspace/configs/", config_name="config")
def main(cfg: DictConfig):
    with logger.profile():
        run(cfg=cfg)


if __name__ == "__main__":
    main()
