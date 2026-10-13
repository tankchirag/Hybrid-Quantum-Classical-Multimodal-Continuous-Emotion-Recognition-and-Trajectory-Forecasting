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

Supported model output formats:
    1) Baseline B1 (dict):
        {
            "va_pred":      (B, T_align, 2),
            "va_future":    (B, S, 2),
            "va_hist":      (B, T_align, 2),
            "va_future_gt": (B, S, 2),
        }

    2) QSAL (tuple):
        (rec_pred_seq, for_pred)
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
        """
        Parameters
        ----------
        model : nn.Module
            BaselineB1, QSAL, or hybrid model.
        optimizer : torch.optim.Optimizer
            Optimizer instance.
        device : str
            "cpu", "cuda", or torch.device.
        patience : int
            Early stopping patience (epochs without improvement).
        save_path : str or None
            Where to save best model state_dict.
        use_tqdm : bool
            Whether to show tqdm progress bars.
        log_batches : bool
            Whether to log per-batch stats in tqdm.
        """
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
        """
        Train for a single epoch.

        Returns
        -------
        dict
            {
                "loss": float,
                "rec_loss": float,
                "fore_loss": float,
            }
        """
        self.model.train()

        total_loss = 0.0
        total_rec = 0.0
        total_fore = 0.0
        batches = 0

        iterator = train_loader
        if self.use_tqdm:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} | Train")

        for batch in iterator:
            batch_start = time.time()

            # Move tensors to device
            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch)

            # ---------------------------------------------
            # SUPPORT BOTH MODEL OUTPUT FORMATS
            # ---------------------------------------------
            if isinstance(outputs, dict):
                # Baseline B1 format
                va_pred = outputs["va_pred"]          # (B, T_align, 2)
                va_future_pred = outputs["va_future"] # (B, S, 2)
                va_hist = outputs["va_hist"]          # (B, T_align, 2)
                va_future_gt = outputs["va_future_gt"]# (B, S, 2)
            else:
                # QSAL format: (rec_pred_seq, for_pred)
                rec_pred_seq, va_future_pred = outputs
                va_pred = rec_pred_seq
                # For QSAL, we assume batch provides aligned targets:
                va_hist = batch["va_hist"]
                va_future_gt = batch["va_future"]

            # Recognition loss (history)
            rec_loss = self.criterion(va_pred, va_hist)

            # Forecasting loss (future)
            fore_loss = self.criterion(va_future_pred, va_future_gt)

            loss = rec_loss + fore_loss

            # Backprop
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
        """
        Evaluate for a single epoch.

        Also handles early stopping and checkpoint saving.

        Returns
        -------
        dict
            {
                "loss": float,
                "rec_loss": float,
                "fore_loss": float,
            }
        """
        self.model.eval()

        total_loss = 0.0
        total_rec = 0.0
        total_fore = 0.0
        batches = 0

        iterator = val_loader
        if self.use_tqdm:
            iterator = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} | Val")

        for batch in iterator:
            # Move tensors to device
            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

            # Forward pass
            outputs = self.model(batch)

            if isinstance(outputs, dict):
                va_pred = outputs["va_pred"]
                va_future_pred = outputs["va_future"]
                va_hist = outputs["va_hist"]
                va_future_gt = outputs["va_future_gt"]
            else:
                rec_pred_seq, va_future_pred = outputs
                va_pred = rec_pred_seq
                va_hist = batch["va_hist"]
                va_future_gt = batch["va_future"]

            rec_loss = self.criterion(va_pred, va_hist)
            fore_loss = self.criterion(va_future_pred, va_future_gt)
            loss = rec_loss + fore_loss

            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_fore += fore_loss.item()
            batches += 1

        avg_loss = total_loss / batches

        # EARLY STOPPING + CHECKPOINT
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.wait = 0

            if self.save_path is not None:
                torch.save(self.model.state_dict(), self.save_path)
        else:
            self.wait += 1

        return {
            "loss": avg_loss,
            "rec_loss": total_rec / batches,
            "fore_loss": total_fore / batches,
        }
