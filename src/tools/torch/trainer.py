import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_fn(
    cfg: DictConfig,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    wandb_logger: wandb,
    total_step: int,
):
    enable_amp = cfg.experiment.fp16
    gradient_accumulation_steps = cfg.experiment.gradient_accumulation_steps
    clip_grad_norm = cfg.experiment.clip_grad_norm
    batch_scheduler = cfg.experiment.batch_scheduler

    scaler = GradScaler(enabled=enable_amp)
    model.to(device)
    model.train()

    total_loss = 0.0
    batch_count = 0
    iteration_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, batch in iteration_bar:
        for k, v in batch.items():
            batch[k] = v.to(device)

        with autocast(enabled=enable_amp):
            batch_outputs = model(batch)
            loss = criterion(batch_outputs, batch["targets"]).item()
            loss = torch.tensor(loss, device=device) / gradient_accumulation_steps

        loss_value = loss.item()
        scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_step += 1

            if batch_scheduler:
                scheduler.step()

        total_loss += loss_value * batch["targets"].size(0)
        batch_count += batch["targets"].size(0)

        ave_loss = total_loss / batch_count
        lr = scheduler.get_last_lr()[0]
        iteration_bar.set_description(f"step: {total_step}, loss: {ave_loss:.4f} lr: {lr:.6f}")
        wandb_logger.log(
            {
                "train_ave_loss": ave_loss,
                "train_loss": loss_value,
                "lr": lr,
                "train_step": total_step,
            }
        )

    if not batch_scheduler:
        scheduler.step()

    return {"loss": total_loss / batch_count, "step": total_step}


def valid_fn(
    cfg: DictConfig,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
):
    gradient_accumulation_steps = cfg.experiment.gradient_accumulation_steps

    outputs = []
    total_loss = 0.0
    batch_count = 0

    model.to(device)
    model.eval()
    iteration_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for _, batch in iteration_bar:
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            batch_outputs = model(batch)
            loss = criterion(batch_outputs, batch["targets"])
            loss = torch.tensor(loss, device=device) / gradient_accumulation_steps

        loss_value = loss.item()
        batch_outputs = batch_outputs.cpu().numpy()

        total_loss += loss_value * batch["targets"].size(0)
        batch_count += batch["targets"].size(0)

        ave_loss = total_loss / batch_count
        iteration_bar.set_description(f"loss: {ave_loss:.4f}")

    outputs = np.concatenate(outputs)
    loss = total_loss / batch_count
    return {"loss": loss, "outputs": outputs}
