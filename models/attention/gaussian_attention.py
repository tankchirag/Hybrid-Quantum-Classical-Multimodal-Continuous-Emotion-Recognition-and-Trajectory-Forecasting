"""
Gaussian Kernel Attention Module for QSAL
-----------------------------------------

This module implements the Gaussian kernel attention mechanism used in the
Quantum Self-Attention Layer (QSAL). It replaces classical dot-product
attention with a distance-based Gaussian kernel:

    alpha_ij = exp( - ||q_i - k_j||^2 / (2 * sigma^2) )

where sigma is a learnable scalar parameter.

This file is placed under:
    models/attention/gaussian_attention.py

to remain consistent with the existing project structure.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianAttention(nn.Module):
    """
    Gaussian kernel attention.

    Given query, key, and value tensors of shape (B, L, D), this module
    computes attention weights using a Gaussian kernel over squared
    Euclidean distances:

        alpha_ij = exp( - ||q_i - k_j||^2 / (2 * sigma^2) )

    The weights are normalized with softmax over the key dimension.

    Parameters
    ----------
    init_sigma : float
        Initial value for sigma (learnable).
    learnable_sigma : bool
        Whether sigma is trainable.
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        init_sigma: float = 1.0,
        learnable_sigma: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.eps = eps

        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(init_sigma)))
            self.log_sigma.requires_grad = False

    # ----------------------------------------------------------------------
    #  Sigma (positive scalar)
    # ----------------------------------------------------------------------

    @property
    def sigma(self) -> torch.Tensor:
        """Return sigma = exp(log_sigma) > 0."""
        return torch.exp(self.log_sigma) + self.eps

    # ----------------------------------------------------------------------
    #  Pairwise squared distances
    # ----------------------------------------------------------------------

    def _pairwise_sq_dist(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise squared Euclidean distances between q and k.

        Parameters
        ----------
        q : torch.Tensor
            Shape (B, L_q, D)
        k : torch.Tensor
            Shape (B, L_k, D)

        Returns
        -------
        torch.Tensor
            Shape (B, L_q, L_k)
        """
        # q: (B, L_q, D) → (B, L_q, 1, D)
        # k: (B, L_k, D) → (B, 1, L_k, D)
        q_exp = q.unsqueeze(2)
        k_exp = k.unsqueeze(1)

        diff = q_exp - k_exp  # (B, L_q, L_k, D)
        dist_sq = torch.sum(diff * diff, dim=-1)  # (B, L_q, L_k)

        return dist_sq

    # ----------------------------------------------------------------------
    #  Forward Pass
    # ----------------------------------------------------------------------

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Gaussian kernel attention.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (B, L_q, D)
        k : torch.Tensor
            Key tensor of shape (B, L_k, D)
        v : torch.Tensor
            Value tensor of shape (B, L_k, D_v)
        mask : torch.Tensor, optional
            Shape (B, L_k) or (B, L_q, L_k). Mask values should be 0 for
            valid positions and -inf for masked positions.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L_q, D_v)
        """
        # 1) Pairwise squared distances
        dist_sq = self._pairwise_sq_dist(q, k)  # (B, L_q, L_k)

        # 2) Gaussian kernel logits
        sigma = self.sigma
        denom = 2.0 * (sigma ** 2)
        logits = -dist_sq / (denom + self.eps)

        # 3) Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # (B, 1, L_k)
            logits = logits + mask

        # 4) Softmax over keys
        attn = F.softmax(logits, dim=-1)  # (B, L_q, L_k)

        # 5) Weighted sum of values
        out = torch.matmul(attn, v)  # (B, L_q, D_v)

        return out
