"""
training.trainer
================

Unified Trainer

Supports

    • Recognition

    • Forecasting

Across

    • Continuous Targets

    • Discrete Targets

Compatible With

    • BaselineB1

    • BaselineB2

    • BaselineB3

    • QSAL

    • MultimodalQEmotionModel

Design Goals
------------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Label Agnostic

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Continuous Labels

✓ Discrete Labels

✓ History Label Compatible

✓ Scheduler Compatible

✓ Early Stopping

✓ Checkpointing

✓ Future Proof
"""

from __future__ import annotations

import time

import torch

from tqdm import tqdm

from training.losses import (
    UnifiedMultiTaskLoss,
)

from training.metrics import (
    evaluate_outputs,
)

from training.scheduler import (
    step_scheduler,
)


class Trainer:
    """
    Unified trainer.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        device="cpu",
        patience=10,
        save_path=None,
        use_tqdm=True,
        log_batches=False,
        alpha=0.2,
        recognition_weight=1.0,
        forecast_weight=1.0,
        gradient_clip=1.0,
        max_train_batches=None,
        max_val_batches=None,
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.device = device

        self.patience = patience

        self.save_path = save_path

        self.use_tqdm = use_tqdm

        self.log_batches = log_batches

        self.gradient_clip = gradient_clip

        self.max_train_batches = max_train_batches

        self.max_val_batches = max_val_batches

        self.loss_fn = UnifiedMultiTaskLoss(
            alpha=alpha,
            recognition_weight=recognition_weight,
            forecast_weight=forecast_weight,
        )

        self.best_val_loss = float("inf")

        self.wait = 0

    # ======================================================
    # Recursive Device Transfer
    # ======================================================

    def _to_device(
        self,
        obj,
    ):

        if torch.is_tensor(obj):

            return obj.to(
                self.device
            )

        if isinstance(
            obj,
            dict,
        ):

            return {

                k: self._to_device(v)

                for k, v in obj.items()
            }

        if isinstance(
            obj,
            list,
        ):

            return [
                self._to_device(v)
                for v in obj
            ]

        return obj

    # ======================================================
    # Train Epoch
    # ======================================================

    def train_one_epoch(
        self,
        train_loader,
        epoch,
        total_epochs,
    ):

        self.model.train()

        total_loss = 0.0

        total_rec = 0.0

        total_fore = 0.0

        num_batches = 0

        iterator = train_loader

        if self.use_tqdm:

            iterator = tqdm(
                train_loader,
                desc=
                f"Epoch {epoch}/{total_epochs} | Train",
            )

        for batch_idx, batch in enumerate(
            iterator
        ):

            if (
                self.max_train_batches
                is not None
                and
                batch_idx >=
                self.max_train_batches
            ):
                break

            batch_start = time.time()

            batch = self._to_device(
                batch
            )

            self.optimizer.zero_grad()

            outputs = self.model(
                batch
            )

            losses = self.loss_fn(

                outputs=outputs,

                recognition_targets=
                batch[
                    "recognition_targets"
                ],

                forecast_targets=
                batch[
                    "forecast_targets"
                ],
            )

            loss = losses[
                "total_loss"
            ]

            # # ==================================================
            # # DEBUG
            # # ==================================================

            # print("\n")
            # print("=" * 60)
            # print("TRAINING DEBUG")
            # print("=" * 60)

            # print(
            #     "loss =",
            #     loss
            # )

            # print(
            #     "loss.requires_grad =",
            #     loss.requires_grad
            # )

            # print(
            #     "loss.grad_fn =",
            #     loss.grad_fn
            # )

            # print()

            # print(
            #     "recognition outputs:"
            # )

            # print(
            #     outputs[
            #         "recognition"
            #     ][
            #         "continuous"
            #     ].keys()
            # )

            # print()

            # print(
            #     "recognition targets:"
            # )

            # print(
            #     batch[
            #         "recognition_targets"
            #     ][
            #         "continuous"
            #     ].keys()
            # )

            # print()

            # print(
            #     "forecast outputs:"
            # )

            # print(
            #     outputs[
            #         "forecasting"
            #     ][
            #         "continuous"
            #     ].keys()
            # )

            # print()

            # print(
            #     "forecast targets:"
            # )

            # print(
            #     batch[
            #         "forecast_targets"
            #     ][
            #         "continuous"
            #     ].keys()
            # )

            # print("=" * 60)
            # print()

            loss.backward()

            if (
                self.gradient_clip
                is not None
            ):

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )

            self.optimizer.step()

            total_loss += float(
                loss.item()
            )

            total_rec += float(
                losses[
                    "recognition_loss"
                ].item()
            )

            total_fore += float(
                losses[
                    "forecasting_loss"
                ].item()
            )

            num_batches += 1

            if (
                self.use_tqdm
                and
                self.log_batches
            ):

                iterator.set_postfix({

                    "loss":
                        f"{loss.item():.4f}",

                    "time":
                        f"{time.time()-batch_start:.2f}s",
                })

        num_batches = max(
            num_batches,
            1,
        )
        return {

            "loss":
                total_loss / num_batches,

            "recognition_loss":
                total_rec / num_batches,

            "forecasting_loss":
                total_fore / num_batches,
        }

    # ======================================================
    # Validation
    # ======================================================

    @torch.no_grad()
    def eval_one_epoch(
        self,
        val_loader,
        epoch,
        total_epochs,
    ):

        self.model.eval()

        total_loss = 0.0

        total_rec = 0.0

        total_fore = 0.0

        num_batches = 0

        collected_metrics = []

        iterator = val_loader

        if self.use_tqdm:

            iterator = tqdm(
                val_loader,
                desc=
                f"Epoch {epoch}/{total_epochs} | Val",
            )

        for batch_idx, batch in enumerate(
            iterator
        ):

            if (
                self.max_val_batches
                is not None
                and
                batch_idx >=
                self.max_val_batches
            ):
                break

            batch = self._to_device(
                batch
            )

            outputs = self.model(
                batch
            )

            losses = self.loss_fn(

                outputs=outputs,

                recognition_targets=
                batch[
                    "recognition_targets"
                ],

                forecast_targets=
                batch[
                    "forecast_targets"
                ],
            )

            metrics = evaluate_outputs(

                outputs=outputs,

                recognition_targets=
                batch[
                    "recognition_targets"
                ],

                forecast_targets=
                batch[
                    "forecast_targets"
                ],
            )

            collected_metrics.append(
                metrics
            )

            total_loss += float(
                losses[
                    "total_loss"
                ].item()
            )

            total_rec += float(
                losses[
                    "recognition_loss"
                ].item()
            )

            total_fore += float(
                losses[
                    "forecasting_loss"
                ].item()
            )

            num_batches += 1

        avg_loss = (
            total_loss
            /
            num_batches
        )

        step_scheduler(
            self.scheduler,
            validation_loss=avg_loss,
        )

        # -----------------------------------
        # Early Stopping
        # -----------------------------------

        if avg_loss < self.best_val_loss:

            self.best_val_loss = avg_loss

            self.wait = 0

            if self.save_path is not None:

                torch.save(
                    self.model.state_dict(),
                    self.save_path,
                )

        else:

            self.wait += 1


        num_batches = max(
            num_batches,
            1,
        )
        
        return {

            "loss":
                avg_loss,

            "recognition_loss":
                total_rec / num_batches,

            "forecasting_loss":
                total_fore / num_batches,

            "metrics":
                collected_metrics,
        }

    # ======================================================
    # Early Stop Flag
    # ======================================================

    @property
    def should_stop(
        self,
    ) -> bool:

        return (
            self.wait
            >=
            self.patience
        )

    # ======================================================
    # Full Fit
    # ======================================================

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
    ):

        history = []

        for epoch in range(
            1,
            epochs + 1,
        ):

            train_stats = (
                self.train_one_epoch(
                    train_loader,
                    epoch,
                    epochs,
                )
            )

            val_stats = (
                self.eval_one_epoch(
                    val_loader,
                    epoch,
                    epochs,
                )
            )

            history.append({

                "epoch":
                    epoch,

                "train":
                    train_stats,

                "val":
                    val_stats,
            })

            print(

                f"[Epoch {epoch}] "

                f"Train={train_stats['loss']:.4f} "

                f"Val={val_stats['loss']:.4f}"

            )

            if self.should_stop:

                print(
                    "Early stopping triggered."
                )

                break

        return history