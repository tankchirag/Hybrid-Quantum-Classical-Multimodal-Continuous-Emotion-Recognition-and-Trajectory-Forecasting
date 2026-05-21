import torch
from training.losses import total_loss


class Trainer:
    def __init__(self, model, optimizer, device="cuda", patience=5):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.patience = patience

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        self.train_history = []
        self.val_history = []

    def train_epoch(self, dataloader):
        self.model.train()
        total = 0.0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            rec_pred, for_pred = self.model(batch)

            loss = total_loss(
                rec_pred,
                batch["va_rec"],
                for_pred,
                batch["va_fut"],
                alpha=0.2,
                lambda_for=1.0,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total += loss.item()

        avg = total / len(dataloader)
        self.train_history.append(avg)
        return avg

    @torch.no_grad()
    def eval_epoch(self, dataloader):
        self.model.eval()
        total = 0.0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            rec_pred, for_pred = self.model(batch)

            loss = total_loss(
                rec_pred,
                batch["va_rec"],
                for_pred,
                batch["va_fut"],
                alpha=0.2,
                lambda_for=1.0,
            )

            total += loss.item()

        avg = total / len(dataloader)
        self.val_history.append(avg)
        return avg

    def fit(self, train_loader, val_loader, epochs=20):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)

            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered.")
                break
