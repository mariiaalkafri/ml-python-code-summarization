import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(path, model, optimizer, epoch, val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    torch.save(state, path)

def run_epoch(model, dataloader, optimizer, criterion, device, train=True, clip_grad=1.0):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    
    for batch in dataloader:
        src_ids = batch.src_ids.to(device)
        src_mask = batch.src_mask.to(device)
        tgt_ids = batch.tgt_ids.to(device)

        # tgt_in: <bos> ... last_token (exclude <eos> or last pad) ?
        # Prompt says: tgt_in = tgt_ids[:, :-1], tgt_out = tgt_ids[:, 1:]
        
        tgt_in = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        if train:
            optimizer.zero_grad()

        # Forward
        # model forward takes (src_ids, src_mask, tgt_in)
        logits = model(src_ids, src_mask, tgt_in)
        
        # Reshape for loss
        # logits: [B, T-1, V] -> [B*(T-1), V]
        # tgt_out: [B, T-1] -> [B*(T-1)]
        
        output_dim = logits.shape[-1]
        loss = criterion(logits.reshape(-1, output_dim), tgt_out.reshape(-1))

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, device, vocab_size, pad_id, 
                epochs=20, lr=3e-4, weight_decay=0.01, save_dir='models'):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=4, min_delta=0.001)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        
        end_time = time.time()
        
        print(f"Epoch {epoch} | Time: {end_time - start_time:.2f}s")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\t Val. Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        # Save last
        save_checkpoint(f"{save_dir}/last.pt", model, optimizer, epoch, val_loss)
        
        # Save best
        if val_loss == early_stopping.best_loss or early_stopping.best_loss is None:
             save_checkpoint(f"{save_dir}/best.pt", model, optimizer, epoch, val_loss)
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
