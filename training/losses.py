import torch
import torch.nn as nn


def ccc_loss(pred, true):
    pred_mean = pred.mean(dim=-2, keepdim=True)
    true_mean = true.mean(dim=-2, keepdim=True)

    pred_var = pred.var(dim=-2, unbiased=False, keepdim=True)
    true_var = true.var(dim=-2, unbiased=False, keepdim=True)

    cov = ((pred - pred_mean) * (true - true_mean)).mean(dim=-2, keepdim=True)

    ccc = 2 * cov / (pred_var + true_var + (pred_mean - true_mean) ** 2 + 1e-8)
    ccc = ccc.mean(dim=-1)

    return 1.0 - ccc.mean()


def mse_loss(pred, true):
    return ((pred - true) ** 2).mean()


def total_loss(rec_pred, rec_true, for_pred, for_true, alpha=0.2, lambda_for=1.0):
    loss_rec = ccc_loss(rec_pred, rec_true) + alpha * mse_loss(rec_pred, rec_true)
    loss_for = ccc_loss(for_pred, for_true) + alpha * mse_loss(for_pred, for_true)

    return loss_rec + lambda_for * loss_for
