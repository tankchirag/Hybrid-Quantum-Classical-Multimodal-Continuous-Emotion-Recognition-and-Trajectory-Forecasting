"""
Universal Trainer for:
    - Baseline B1 (Classical Fusion)
    - QSAL (Quantum Self-Attention Layer)
    - Hybrid Models

Features:
    - tqdm progress bars
    - per-batch timing
    - per-epoch timing
    - early stopping
    - checkpoint saving
    - supports both model output formats:
        * Baseline B1: {"va_pred": (B,T,2), "va_future": (B,S,2)}
        * QSAL: (rec_pred_seq, for_pred)
"""

import time
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device="cpu",
        patience=5,
        save_path=None,
        use_tqdm=True,
        log_batches=False,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.save_path = save_path
        self.use_tqdm = use_tqdm
        self.log_batches = log_batches

        self.criterion = nn.MSELoss()

        self.best_val_loss = float("inf")
        self.wait = 0  # early stopping counter

    # ---------------------------------------------------------
    # TRAIN ONE EPOCH
    # ---------------------------------------------------------
    def train_one_epoch(self, train_loader, epoch, total_epochs):
        self.model.train()

        total_loss = 0
        total_rec = 0
        total_fore = 0
        batches = 0

        iterator = train_loader
        if self.use_tqdm:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} | Train")

        for batch in iterator:
            batch_start = time.time()

            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()

            outputs = self.model(batch)

            # ---------------------------------------------
            # SUPPORT BOTH MODEL OUTPUT FORMATS
            # ---------------------------------------------
            if isinstance(outputs, dict):
                # Baseline B1 format
                va_pred = outputs["va_pred"]          # (B,T,2)
                va_future_pred = outputs["va_future"] # (B,S,2)
            else:
                # QSAL format
                rec_pred_seq, va_future_pred = outputs
                va_pred = rec_pred_seq                # (B,T,2)

            rec_loss = self.criterion(va_pred, batch["va_hist"])
            fore_loss = self.criterion(va_future_pred, batch["va_future"])
            loss = rec_loss + fore_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_fore += fore_loss.item()
            batches += 1

            if self.log_batches and self.use_tqdm:
                iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "batch_time": f"{time.time() - batch_start:.3f}s"
                })

        return {
            "loss": total_loss / batches,
            "rec_loss": total_rec / batches,
            "fore_loss": total_fore / batches,
        }

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------
    @torch.no_grad()
    def eval_one_epoch(self, val_loader, epoch, total_epochs):
        self.model.eval()

        total_loss = 0
        total_rec = 0
        total_fore = 0
        batches = 0

        iterator = val_loader
        if self.use_tqdm:
            iterator = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} | Val")

        for batch in iterator:
            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

            outputs = self.model(batch)

            if isinstance(outputs, dict):
                va_pred = outputs["va_pred"]
                va_future_pred = outputs["va_future"]
            else:
                rec_pred_seq, va_future_pred = outputs
                va_pred = rec_pred_seq

            rec_loss = self.criterion(va_pred, batch["va_hist"])
            fore_loss = self.criterion(va_future_pred, batch["va_future"])
            loss = rec_loss + fore_loss

            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_fore += fore_loss.item()
            batches += 1

        # EARLY STOPPING + CHECKPOINT
        if total_loss < self.best_val_loss:
            self.best_val_loss = total_loss
            self.wait = 0

            if self.save_path is not None:
                torch.save(self.model.state_dict(), self.save_path)
        else:
            self.wait += 1

        return {
            "loss": total_loss / batches,
            "rec_loss": total_rec / batches,
            "fore_loss": total_fore / batches,
        }
