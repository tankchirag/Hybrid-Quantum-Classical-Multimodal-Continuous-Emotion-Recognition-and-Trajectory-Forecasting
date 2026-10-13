"""
Loss functions for multimodal continuous emotion recognition
and trajectory forecasting.

Includes:
    - Concordance Correlation Coefficient (CCC) loss
    - Mean Squared Error (MSE) loss
    - Combined total loss for recognition + forecasting
"""

import torch
import torch.nn as nn


def ccc_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Concordance Correlation Coefficient (CCC) loss.

    Shape-agnostic:
        - Works for (B, 2)
        - Works for (B, T, 2)
        - Works for (B, S, 2)

    Strategy:
        - Flatten all leading dimensions into one:
              pred: (N, 2), true: (N, 2)
          where N = B * T or B * S or B
        - Compute CCC per dimension (valence, arousal)
        - Return 1 - mean(CCC) as loss

    Parameters
    ----------
    pred : Tensor
        Predicted VA values, shape (..., 2).
    true : Tensor
        Ground-truth VA values, shape (..., 2).

    Returns
    -------
    loss : Tensor
        Scalar CCC loss (1 - CCC).
    """
    # Ensure same shape
    assert pred.shape == true.shape, \
        f"ccc_loss: shape mismatch pred {pred.shape} vs true {true.shape}"

    # Flatten all but last dim (VA=2)
    pred = pred.view(-1, pred.size(-1))   # (N, 2)
    true = true.view(-1, true.size(-1))   # (N, 2)

    # Means over N
    pred_mean = pred.mean(dim=0, keepdim=True)   # (1, 2)
    true_mean = true.mean(dim=0, keepdim=True)   # (1, 2)

    # Variances over N
    pred_var = pred.var(dim=0, unbiased=False, keepdim=True)  # (1, 2)
    true_var = true.var(dim=0, unbiased=False, keepdim=True)  # (1, 2)

    # Covariance over N
    cov = ((pred - pred_mean) * (true - true_mean)).mean(dim=0, keepdim=True)  # (1, 2)

    # CCC per dimension
    ccc = 2 * cov / (pred_var + true_var + (pred_mean - true_mean) ** 2 + 1e-8)  # (1, 2)

    # Loss = 1 - mean CCC over VA dims
    loss = 1.0 - ccc.mean()
    return loss


def mse_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Standard Mean Squared Error (MSE) loss.

    Parameters
    ----------
    pred : Tensor
        Predicted values, shape (..., 2).
    true : Tensor
        Ground-truth values, shape (..., 2).

    Returns
    -------
    loss : Tensor
        Scalar MSE loss.
    """
    return nn.functional.mse_loss(pred, true)


def total_loss(
    rec_pred: torch.Tensor,
    rec_true: torch.Tensor,
    for_pred: torch.Tensor,
    for_true: torch.Tensor,
    alpha: float = 0.2,
    lambda_for: float = 1.0,
) -> torch.Tensor:
    """
    Combined loss for recognition + forecasting.

    Recognition:
        - rec_pred, rec_true: (..., 2)
        - Uses CCC + alpha * MSE

    Forecasting:
        - for_pred, for_true: (..., 2)
        - Uses CCC + alpha * MSE, scaled by lambda_for

    Parameters
    ----------
    rec_pred : Tensor
        Recognition predictions (VA at t).
    rec_true : Tensor
        Recognition ground truth (VA at t).
    for_pred : Tensor
        Forecasting predictions (VA at t+1...t+S).
    for_true : Tensor
        Forecasting ground truth (VA at t+1...t+S).
    alpha : float
        Weight for MSE term in both recognition and forecasting.
    lambda_for : float
        Weight for the forecasting loss block.

    Returns
    -------
    loss : Tensor
        Scalar total loss.
    """
    # Recognition loss
    loss_rec = ccc_loss(rec_pred, rec_true) + alpha * mse_loss(rec_pred, rec_true)

    # Forecasting loss
    loss_for = ccc_loss(for_pred, for_true) + alpha * mse_loss(for_pred, for_true)

    return loss_rec + lambda_for * loss_for
