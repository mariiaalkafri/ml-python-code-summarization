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


def save_checkpoint(path, model, optimizer, epoch, val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    torch.save(state, path)


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    """
    logits: [B, T, V]
    targets: [B, T]
    """
    preds = logits.argmax(dim=-1)
    mask = targets.ne(pad_id)
    correct = (preds.eq(targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


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
    n_steps = 0

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
            tgt_out.reshape(-1),
        )

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        # accuracy
        with torch.no_grad():
            acc = token_accuracy(logits, tgt_out, pad_id)

        total_loss += loss.item()
        total_acc += acc
        n_steps += 1

        if train and (i % log_every == 0):
            print(f"  batch {i}/{len(dataloader)}  loss={loss.item():.4f}  acc={acc*100:.2f}%")

    return (total_loss / max(1, n_steps)), (total_acc / max(1, n_steps))


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    pad_id,
    epochs=10,
    lr=3e-4,
    weight_decay=0.01,
    save_dir="models",
    log_every=200,
    clip_grad=1.0,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    early_stopping = EarlyStopping(patience=4, min_delta=0.001)
    best_val = float("inf")

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = run_epoch(
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

        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pad_id=pad_id,
            train=False,
        )

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch} | Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} |   Val Acc: {val_acc*100:.2f}%")

        scheduler.step(val_loss)

        # Save last every epoch
        save_checkpoint(os.path.join(save_dir, "last.pt"), model, optimizer, epoch, val_loss)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(save_dir, "best.pt"), model, optimizer, epoch, val_loss)
            print("  ✔ Saved new best model")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

