"""
Trainer Module (FAST DEBUG MODE)

Enhancements over the standard trainer:
    - tqdm progress bars for training and validation
    - Per-batch timing logs
    - Per-epoch timing logs
    - Debug-friendly printouts
    - No change to scientific architecture or loss logic

This preserves the thesis-correct behavior:
    Recognition:
        rec_pred_seq : (B, T, 2)
        rec_pred     : last time step → (B, 2)

    Forecasting:
        for_pred     : (B, S, 2)
"""

import time
import torch
from tqdm import tqdm
from training.losses import total_loss


class Trainer:
    """
    Trainer class for multimodal quantum-classical emotion recognition
    and trajectory forecasting.

    Parameters
    ----------
    model : nn.Module
        The multimodal QSAL + Transformer model.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    device : str
        "cuda" or "cpu".
    patience : int
        Early stopping patience.
    debug_mode : bool
        If True → enables progress bars + timing logs.
    """

    def __init__(self, model, optimizer, device="cpu", patience=5, debug_mode=True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.debug_mode = debug_mode

        # Early stopping state
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        # Logging
        self.train_history = []
        self.val_history = []

    # ---------------------------------------------------------
    # TRAINING EPOCH
    # ---------------------------------------------------------
    def train_epoch(self, dataloader):
        """
        One full training epoch.

        Debug mode adds:
            - tqdm progress bar
            - per-batch timing
            - per-epoch timing
        """
        self.model.train()
        total = 0.0

        epoch_start = time.time()

        iterator = (
            tqdm(dataloader, desc="Training", ncols=100)
            if self.debug_mode
            else dataloader
        )

        for batch in iterator:
            batch_start = time.time()

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            rec_pred_seq, for_pred = self.model(batch)

            # Recognition: last time step
            rec_pred = rec_pred_seq[:, -1, :]
            rec_true = batch["va_rec"]

            # Forecasting: full sequence
            for_true = batch["va_fut"]

            # Combined loss
            loss = total_loss(
                rec_pred=rec_pred,
                rec_true=rec_true,
                for_pred=for_pred,
                for_true=for_true,
                alpha=0.2,
                lambda_for=1.0,
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total += loss.item()

            # Debug timing
            if self.debug_mode:
                batch_time = time.time() - batch_start
                iterator.set_postfix({
                    "batch_time": f"{batch_time:.3f}s",
                    "loss": f"{loss.item():.4f}"
                })

        epoch_time = time.time() - epoch_start
        if self.debug_mode:
            print(f"Epoch train time: {epoch_time:.2f}s")

        avg = total / len(dataloader)
        self.train_history.append(avg)
        return avg

    # ---------------------------------------------------------
    # VALIDATION EPOCH
    # ---------------------------------------------------------
    @torch.no_grad()
    def eval_epoch(self, dataloader):
        """
        One full validation epoch.

        Debug mode adds:
            - tqdm progress bar
            - per-epoch timing
        """
        self.model.eval()
        total = 0.0

        epoch_start = time.time()

        iterator = (
            tqdm(dataloader, desc="Validation", ncols=100)
            if self.debug_mode
            else dataloader
        )

        for batch in iterator:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            rec_pred_seq, for_pred = self.model(batch)

            # Recognition: last time step
            rec_pred = rec_pred_seq[:, -1, :]
            rec_true = batch["va_rec"]

            # Forecasting: full sequence
            for_true = batch["va_fut"]

            loss = total_loss(
                rec_pred=rec_pred,
                rec_true=rec_true,
                for_pred=for_pred,
                for_true=for_true,
                alpha=0.2,
                lambda_for=1.0,
            )

            total += loss.item()

        epoch_time = time.time() - epoch_start
        if self.debug_mode:
            print(f"Epoch val time: {epoch_time:.2f}s")

        avg = total / len(dataloader)
        self.val_history.append(avg)
        return avg

    # ---------------------------------------------------------
    # FIT LOOP WITH EARLY STOPPING
    # ---------------------------------------------------------
    def fit(self, train_loader, val_loader, epochs=20):
        """
        Full training loop with early stopping.

        Saves best model to:
            best_model.pt
        """
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch+1}/{epochs} =====")

            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)

            print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered.")
                break
