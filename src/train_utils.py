# src/train_utils.py
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class EpochStats:
    loss: float
    tok_acc: float  # token accuracy ignoring pad


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return False

        improved = (self.best_loss - val_loss) > self.min_delta
        if improved:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    """
    logits:  [B, T, V]
    targets: [B, T]
    Accuracy ignores pad tokens.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)      # [B, T]
        mask = (targets != pad_id)         # [B, T]
        denom = mask.sum().item()
        if denom == 0:
            return 0.0
        correct = ((preds == targets) & mask).sum().item()
        return correct / denom


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer,
                    scheduler, epoch: int, best_val_loss: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer,
                    scheduler, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", 1e9))
    return start_epoch, best_val_loss


def run_epoch(
    model: nn.Module,
    dataloader,
    optimizer: Optional[optim.Optimizer],
    criterion: nn.Module,
    device: str,
    pad_id: int,
    train: bool,
    clip_grad: float,
    log_every: int,
) -> EpochStats:
    model.train() if train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        src_ids = batch.src_ids.to(device, non_blocking=True)
        src_mask = batch.src_mask.to(device, non_blocking=True)
        tgt_ids = batch.tgt_ids.to(device, non_blocking=True)

        tgt_in = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(src_ids, src_mask, tgt_in)  # [B, T, V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            acc = token_accuracy(logits, tgt_out, pad_id)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

        total_loss += float(loss.item())
        total_acc += float(acc)

        if train and (i % log_every == 0):
            print(f"  batch {i}/{n_batches}  loss={loss.item():.4f}  acc={acc*100:.2f}%")

    return EpochStats(
        loss=total_loss / max(1, n_batches),
        tok_acc=total_acc / max(1, n_batches),
    )


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    pad_id: int,
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    save_dir: str = "models",
    log_every: int = 200,
    clip_grad: float = 1.0,
    patience: int = 3,
    min_delta: float = 0.0,
    resume: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pt")
    last_path = os.path.join(save_dir, "last.pt")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    early = EarlyStopping(patience=patience, min_delta=min_delta)

    start_epoch = 1
    best_val = 1e9

    if resume and os.path.exists(last_path):
        print(f"Resuming from {last_path}")
        start_epoch, best_val = load_checkpoint(last_path, model, optimizer, scheduler, device)
        early.best_loss = best_val
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val:.4f}")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_stats = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pad_id=pad_id,
            train=True,
            clip_grad=clip_grad,
            log_every=log_every,
        )

        val_stats = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            pad_id=pad_id,
            train=False,
            clip_grad=clip_grad,
            log_every=max(1, log_every),
        )

        dt = time.time() - t0

        print(f"\nEpoch {epoch} | Time: {dt:.2f}s")
        print(f"  Train Loss: {train_stats.loss:.4f} | Train Acc: {train_stats.tok_acc*100:.2f}%")
        print(f"    Val Loss: {val_stats.loss:.4f} |   Val Acc: {val_stats.tok_acc*100:.2f}%")

        scheduler.step(val_stats.loss)

        # Always save last
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val)

        # Save best if improved
        if val_stats.loss < best_val:
            best_val = val_stats.loss
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val)
            print("  ✔ Saved new best model")

        if early.step(val_stats.loss):
            print("  ⛔ Early stopping triggered")
            break

        print("-" * 60)

    print("Training finished.")
    print("Best:", best_path)
    print("Last:", last_path)
