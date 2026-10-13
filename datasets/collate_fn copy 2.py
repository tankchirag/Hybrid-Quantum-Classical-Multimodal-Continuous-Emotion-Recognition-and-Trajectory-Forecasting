"""
datasets.collate_fn
===================

Universal Multimodal Collate Function

This module provides a fully:

    ✓ modality-agnostic
    ✓ dataset-agnostic
    ✓ label-agnostic
    ✓ forecasting-compatible
    ✓ configurable

batch construction utility.

Supported datasets
------------------

    • RECOLA
    • SEWA
    • ULM-TSST Synthetic
    • Future custom datasets

Motivation
----------

Different datasets expose different modalities:

RECOLA

    audio
    video
    ecg
    eda

SEWA

    audio
    video
    text

ULM-TSST

    audio
    video
    ecg
    eda
    eeg
    emg
    resp
    bpm
    eye
    text

Future datasets may contain:

    radar
    imu
    ppg
    thermal
    gps
    accelerometer
    ...

The collate function should never require code changes
when a new modality or label is introduced.

Examples
--------

Input sample:

{
    "modalities": {
        "audio": Tensor(T,1),
        "ecg": Tensor(T,1)
    },

    "labels": {
        "valence": Tensor(T),
        "arousal": Tensor(T)
    },

    "metadata": {...}
}

Output:

{
    "modalities": {
        "audio": Tensor(B,T,1),
        "ecg": Tensor(B,T,1)
    },

    "labels": {
        "valence": Tensor(B,T),
        "arousal": Tensor(B,T)
    },

    "metadata": {...}
}
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

import torch


# ==========================================================
# Helper Functions
# ==========================================================

def _is_tensor_batch(
    values: List[Any],
) -> bool:
    """
    Check whether all elements are tensors.
    """

    return all(
        isinstance(v, torch.Tensor)
        for v in values
    )


# ==========================================================
# Main Collate Function
# ==========================================================

def multimodal_collate_fn(
    batch: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Universal multimodal collate function.

    Parameters
    ----------
    batch : list of dict

        List of dataset samples.

    Returns
    -------
    dict

        {
            "modalities": {...},
            "labels": {...},
            "metadata": {...}
        }
    """

    if len(batch) == 0:
        raise ValueError(
            "Received empty batch."
        )

    batch_size = len(batch)

    output = {
        "modalities": {},
        "labels": {},
        "metadata": {},
    }

    # =====================================================
    # Discover all modalities dynamically
    # =====================================================

    modality_keys = set()

    for sample in batch:
        modality_keys.update(
            sample.get(
                "modalities",
                {},
            ).keys()
        )

    # =====================================================
    # Collate Modalities
    # =====================================================

    for modality in modality_keys:

        values = []

        for sample in batch:

            value = (
                sample
                .get("modalities", {})
                .get(modality, None)
            )

            values.append(value)

        available = [
            v for v in values
            if v is not None
        ]

        if len(available) == 0:
            output["modalities"][
                modality
            ] = None
            continue

        first = available[0]

        # ---------------------------------------------
        # Tensor modalities
        # ---------------------------------------------

        if isinstance(first, torch.Tensor):

            fill_tensor = torch.zeros_like(
                first
            )

            stacked = torch.stack([
                v
                if v is not None
                else fill_tensor
                for v in values
            ])

            output["modalities"][
                modality
            ] = stacked

        # ---------------------------------------------
        # Text / Strings
        # ---------------------------------------------

        elif isinstance(first, str):

            output["modalities"][
                modality
            ] = [
                v if v is not None else ""
                for v in values
            ]

        # ---------------------------------------------
        # Generic objects
        # ---------------------------------------------

        else:

            output["modalities"][
                modality
            ] = values

    # =====================================================
    # Discover labels dynamically
    # =====================================================

    label_keys = set()

    for sample in batch:

        label_keys.update(
            sample.get(
                "labels",
                {},
            ).keys()
        )

    # =====================================================
    # Collate Labels
    # =====================================================

    for label in label_keys:

        values = []

        for sample in batch:

            value = (
                sample
                .get("labels", {})
                .get(label, None)
            )

            values.append(value)

        available = [
            v for v in values
            if v is not None
        ]

        if len(available) == 0:

            output["labels"][
                label
            ] = None

            continue

        first = available[0]

        if isinstance(
            first,
            torch.Tensor,
        ):

            fill_tensor = torch.zeros_like(
                first
            )

            output["labels"][
                label
            ] = torch.stack([
                v
                if v is not None
                else fill_tensor
                for v in values
            ])

        else:

            output["labels"][
                label
            ] = values

    # =====================================================
    # Metadata
    # =====================================================

    metadata_keys = set()

    for sample in batch:

        metadata_keys.update(
            sample.get(
                "metadata",
                {},
            ).keys()
        )

    for key in metadata_keys:

        output["metadata"][
            key
        ] = [

            sample
            .get("metadata", {})
            .get(key)

            for sample in batch
        ]

    # =====================================================
    # Useful runtime information
    # =====================================================

    output["batch_size"] = batch_size

    output["active_modalities"] = list(
        modality_keys
    )

    output["active_labels"] = list(
        label_keys
    )

    return output