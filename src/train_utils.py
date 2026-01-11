import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": val_loss,
        "best_val": best_val,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    # model
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # optimizer
    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # scheduler
    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", 0)  # last completed epoch
    best_val = ckpt.get("best_val", float("inf"))
    return start_epoch, best_val


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    """
    logits: [B, T, V]
    targets: [B, T]
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)  # [B, T]
        mask = (targets != pad_id)
        correct = (preds == targets) & mask
        denom = mask.sum().item()
        if denom == 0:
            return 0.0
        return (correct.sum().item() / denom) * 100.0


def run_epoch(model, dataloader, optimizer, criterion, device,
              pad_id: int, train=True, clip_grad=1.0, log_every=200):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for i, batch in enumerate(dataloader):
        src_ids = batch.src_ids.to(device, non_blocking=True)
        src_mask = batch.src_mask.to(device, non_blocking=True)
        tgt_ids = batch.tgt_ids.to(device, non_blocking=True)

        tgt_in = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        if train:
            optimizer.zero_grad()
            logits = model(src_ids, src_mask, tgt_in)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(src_ids, src_mask, tgt_in)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1)
                )

        acc = token_accuracy(logits, tgt_out, pad_id)

        total_loss += loss.item()
        total_acc += acc

        if train and (i % log_every == 0):
            print(f"  batch {i}/{len(dataloader)}  loss={loss.item():.4f}  acc={acc:.2f}%")

    avg_loss = total_loss / max(1, len(dataloader))
    avg_acc = total_acc / max(1, len(dataloader))
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, device, pad_id,
                epochs_total=20, lr=3e-4, weight_decay=0.01,
                save_dir="models", resume_path=None,
                log_every=200, clip_grad=1.0):

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    early_stopping = EarlyStopping(patience=4, min_delta=0.001)

    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    best_val = float("inf")

    # --- Resume ---
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        start_epoch, best_val = load_checkpoint(
            resume_path, model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        print(f"Checkpoint loaded. Last completed epoch: {start_epoch}. Best val so far: {best_val:.4f}")
    else:
        if resume_path:
            print(f"Resume checkpoint not found at: {resume_path} (starting fresh)")

    # Continue from next epoch
    for epoch in range(start_epoch + 1, epochs_total + 1):
        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, device,
            pad_id=pad_id, train=True, clip_grad=clip_grad, log_every=log_every
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion, device,
            pad_id=pad_id, train=False
        )

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch} | Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} |   Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # Save last
        save_checkpoint(
            os.path.join(save_dir, "last.pt"),
            model, optimizer, scheduler, epoch, val_loss, best_val
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                os.path.join(save_dir, "best.pt"),
                model, optimizer, scheduler, epoch, val_loss, best_val
            )
            print("  ✔ Saved new best model")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
