# src/train_utils.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------
# Metrics
# -------------------------
def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    """
    Token-level accuracy ignoring pad tokens.
    logits: [B, T, V]
    targets: [B, T]
    """
    preds = logits.argmax(dim=-1)  # [B, T]
    mask = targets != pad_id
    correct = (preds == targets) & mask

    correct_tokens = correct.sum().item()
    total_tokens = mask.sum().item()

    if total_tokens == 0:
        return 0.0

    return correct_tokens / total_tokens


# -------------------------
# Checkpointing
# -------------------------
def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    return start_epoch, best_val_loss


# -------------------------
# One epoch
# -------------------------
def run_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    pad_id,
    train=True,
    clip_grad=1.0,
    log_every=200,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        src_ids = batch.src_ids.to(device)
        src_mask = batch.src_mask.to(device)
        tgt_ids = batch.tgt_ids.to(device)

        tgt_in = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(src_ids, src_mask, tgt_in)  # [B, T-1, V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            if train:
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

        acc = token_accuracy(logits, tgt_out, pad_id)

        total_loss += loss.item()
        total_acc += acc

        if train and (i % log_every == 0):
            elapsed = time.time() - start_time
            print(
                f"  batch {i}/{len(dataloader)}  "
                f"loss={loss.item():.4f}  acc={acc:.4f}  time={elapsed:.2f}s"
            )

    return total_loss / max(1, len(dataloader)), total_acc / max(1, len(dataloader))


# -------------------------
# Full training
# -------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    pad_id,
    epochs=10,
    lr=3e-4,
    weight_decay=0.01,
    clip_grad=1.0,
    log_every=200,
    save_dir="models",
    resume=True,
):
    best_path = os.path.join(save_dir, "best.pt")
    last_path = os.path.join(save_dir, "last.pt")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume if last checkpoint exists
    if resume and os.path.exists(last_path):
        try:
            print(f"Found checkpoint {last_path} — resuming...")
            start_epoch, best_val_loss = load_checkpoint(last_path, model, optimizer, device)
            print(f"Resumed from epoch {start_epoch} | best_val_loss={best_val_loss:.4f}")
        except Exception as e:
            print("Could not resume, starting fresh. Reason:", e)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            pad_id,
            train=True,
            clip_grad=clip_grad,
            log_every=log_every,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            pad_id,
            train=False,
            clip_grad=clip_grad,
            log_every=log_every,
        )

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f} |  Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Save last always
        save_checkpoint(last_path, model, optimizer, epoch, best_val_loss)

        # Save best if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_val_loss)
            print("  ✔ Saved new best model")

        print("-" * 60)

    print("Training finished.")
    print("Best model:", best_path)
    print("Last checkpoint:", last_path)
