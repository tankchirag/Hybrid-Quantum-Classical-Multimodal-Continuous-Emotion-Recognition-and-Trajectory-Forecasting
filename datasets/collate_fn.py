"""
collate_fn.py
=============

Universal Multimodal Collate Function

Creates training batches from WindowSample
objects.

Design Goals
------------

✓ Modality Agnostic

✓ Label Agnostic

✓ Dataset Agnostic

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Continuous Labels

✓ Discrete Labels

✓ Continuous Forecasting

✓ Discrete Forecasting

✓ History Label Support

✓ Future Proof

Batch Structure
---------------

{
    "modalities": {...},

    "continuous_labels": {...},

    "discrete_labels": {...},

    "recognition_targets": {

        "continuous": {...},

        "discrete": {...}
    },

    "forecast_targets": {

        "continuous": {...},

        "discrete": {...}
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
# Tensor Dictionary Stacking
# ==========================================================

def _stack_dictionary_tensors(
    values: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Stack dictionary tensors.

    Input

        [
            {"x": T1},
            {"x": T2}
        ]

    Output

        {
            "x": stack(...)
        }
    """

    if len(values) == 0:
        return {}

    keys = set()

    for item in values:
        keys.update(item.keys())

    output = {}

    for key in keys:

        tensors = []

        for item in values:

            if key in item:
                tensors.append(item[key])

        if len(tensors) > 0:

            output[key] = torch.stack(
                tensors,
                dim=0,
            )

    return output


# ==========================================================
# Nested Target Stacking
# ==========================================================

def _stack_target_dictionary(
    values: List[Dict[str, Dict[str, torch.Tensor]]],
):
    """
    Stack recognition/forecast target dictionaries.

    Example

    Input

        [
            {
                "continuous": {
                    "valence": ...
                },

                "discrete": {
                    "emotion_class": ...
                }
            }
        ]

    Output

        {
            "continuous": {
                "valence": (B,...)
            },

            "discrete": {
                "emotion_class": (B,...)
            }
        }
    """

    output = {

        "continuous": {},

        "discrete": {},
    }

    # ------------------------------------------
    # Continuous
    # ------------------------------------------

    continuous_items = [

        item.get(
            "continuous",
            {},
        )

        for item in values
    ]

    output[
        "continuous"
    ] = _stack_dictionary_tensors(
        continuous_items
    )

    # ------------------------------------------
    # Discrete
    # ------------------------------------------

    discrete_items = [

        item.get(
            "discrete",
            {},
        )

        for item in values
    ]

    output[
        "discrete"
    ] = _stack_dictionary_tensors(
        discrete_items
    )

    return output


# ==========================================================
# Main Collate Function
# ==========================================================

def multimodal_collate_fn(
    batch: List[Any],
) -> Dict[str, Any]:
    """
    Universal batch builder.
    """

    if len(batch) == 0:

        raise ValueError(
            "Received empty batch."
        )

    # ------------------------------------------------------
    # Modalities
    # ------------------------------------------------------

    modality_batch = (
        _stack_dictionary_tensors(
            [
                sample.modalities
                for sample in batch
            ]
        )
    )

    # ------------------------------------------------------
    # Continuous Histories
    # ------------------------------------------------------

    continuous_batch = (
        _stack_dictionary_tensors(
            [
                sample.continuous_labels
                for sample in batch
            ]
        )
    )

    # ------------------------------------------------------
    # Discrete Histories
    # ------------------------------------------------------

    discrete_batch = (
        _stack_dictionary_tensors(
            [
                sample.discrete_labels
                for sample in batch
            ]
        )
    )

    # ------------------------------------------------------
    # Recognition Targets
    # ------------------------------------------------------

    recognition_batch = (
        _stack_target_dictionary(
            [
                sample.recognition_targets
                for sample in batch
            ]
        )
    )

    # ------------------------------------------------------
    # Forecast Targets
    # ------------------------------------------------------

    forecast_batch = (
        _stack_target_dictionary(
            [
                sample.forecast_targets
                for sample in batch
            ]
        )
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    metadata = [

        sample.metadata

        for sample in batch
    ]

    # ------------------------------------------------------
    # Active Modalities
    # ------------------------------------------------------

    active_modalities = list(
        modality_batch.keys()
    )

    active_continuous = list(
        continuous_batch.keys()
    )

    active_discrete = list(
        discrete_batch.keys()
    )

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    return {

        "batch_size":
            len(batch),

        "active_modalities":
            active_modalities,

        "active_continuous_labels":
            active_continuous,

        "active_discrete_labels":
            active_discrete,

        "modalities":
            modality_batch,

        "continuous_labels":
            continuous_batch,

        "discrete_labels":
            discrete_batch,

        "recognition_targets":
            recognition_batch,

        "forecast_targets":
            forecast_batch,

        "metadata":
            metadata,
    }