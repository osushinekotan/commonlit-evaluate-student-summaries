import gc
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.tools.torch.trainer import train_fn, valid_fn
from src.utils.logger import Logger

logger = Logger(__name__)

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


def train_loop(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Training loop for a given experiment configuration, training and validation data."""

    # Instantiate datasets and dataloaders.
    train_dataset = instantiate(cfg.experiment.dataset, cfg=cfg, df=train_df)
    valid_dataset = instantiate(cfg.experiment.dataset, cfg=cfg, df=valid_df)
    train_loader = instantiate(cfg.experiment.train_dataloader, dataset=train_dataset)
    valid_loader = instantiate(cfg.experiment.valid_dataloader, dataset=valid_dataset)

    # Instantiate the model, optimizer, scheduler, metrics, and loss criterion.
    model = instantiate(cfg.experiment.model, cfg=cfg)
    optimizer = instantiate(cfg.experiment.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.experiment.scheduler, optimizer=optimizer)
    metrics = instantiate(cfg.experiment.metrics)
    criterion = instantiate(cfg.experiment.criterion)
    max_epochs = instantiate(cfg.experiment.max_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_steps = 0  # Track total steps across all epochs.
    best_score = -np.inf  # Initialize the best score to negative infinity.

    # Start the training loop.
    for epoch in range(max_epochs):
        # Train for one epoch and get training loss and step.
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
        eval_results = metrics(outputs=validation_output, targets=valid_df["targets"])

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
            torch.save(model.state_dict(), output_dir / "model.pth")  # Save the model's state dict.
            joblib.dump(validation_output, output_dir / "output.pkl")  # Save the validation outputs.

    wandb.finish(quiet=True)
    torch.cuda.empty_cache()
    gc.collect()

    # Load and return the best validation outputs.
    best_validation_outputs = joblib.load(output_dir / "output.pkl")
    return best_validation_outputs
