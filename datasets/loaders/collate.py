"""
datasets.loaders.collate
========================

Universal Batch Collate Function

Purpose
-------
Convert WindowSample objects into
training-ready batches.

Design Goals
------------

✓ Dataset Agnostic

✓ Modality Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition

✓ Forecasting

✓ Multi-task Learning

✓ Future Proof

Input
-----

List[WindowSample]

Output
------

{
    "batch_size": int,

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

    "metadata": [...]
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

def _stack_tensor_dictionaries(
    items: List[
        Dict[str, torch.Tensor]
    ],
) -> Dict[str, torch.Tensor]:
    """
    Stack dictionary tensors.

    Example
    -------

    [
        {"ecg": T1},
        {"ecg": T2}
    ]

    →

    {
        "ecg": stack(...)
    }
    """

    output = {}

    if len(items) == 0:
        return output

    keys = set()

    for item in items:

        keys.update(
            item.keys()
        )

    for key in sorted(keys):

        tensors = []

        for item in items:

            if key in item:

                tensors.append(
                    item[key]
                )

        if len(tensors) == 0:
            continue

        output[key] = torch.stack(
            tensors,
            dim=0,
        )

    return output


# ==========================================================
# Nested Targets
# ==========================================================

def _stack_nested_targets(
    targets,
):

    return {

        "continuous":

        _stack_tensor_dictionaries(

            [
                target.get(
                    "continuous",
                    {},
                )

                for target
                in targets
            ]
        ),

        "discrete":

        _stack_tensor_dictionaries(

            [
                target.get(
                    "discrete",
                    {},
                )

                for target
                in targets
            ]
        ),
    }


# ==========================================================
# Main Collate
# ==========================================================

def collate(
    batch: List[Any],
) -> Dict[str, Any]:
    """
    Universal batch builder.

    Assumptions
    -----------

    Input samples already passed:

        DiskDataset

            ↓

        TimelineSynchronizer

            ↓

        WindowSampler
    """

    if len(batch) == 0:

        raise ValueError(
            "Empty batch received."
        )

    # ------------------------------------------------------
    # Histories
    # ------------------------------------------------------

    modalities = (

        _stack_tensor_dictionaries(

            [
                sample.modalities

                for sample
                in batch
            ]
        )
    )

    continuous_labels = (

        _stack_tensor_dictionaries(

            [
                sample.continuous_labels

                for sample
                in batch
            ]
        )
    )

    discrete_labels = (

        _stack_tensor_dictionaries(

            [
                sample.discrete_labels

                for sample
                in batch
            ]
        )
    )

    # ------------------------------------------------------
    # Targets
    # ------------------------------------------------------

    recognition_targets = (

        _stack_nested_targets(

            [
                sample
                .recognition_targets

                for sample
                in batch
            ]
        )
    )

    forecast_targets = (

        _stack_nested_targets(

            [
                sample
                .forecast_targets

                for sample
                in batch
            ]
        )
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    metadata = [

        sample.metadata

        for sample
        in batch
    ]

    # ------------------------------------------------------
    # Dynamic Discovery
    # ------------------------------------------------------

    active_modalities = sorted(
        modalities.keys()
    )

    active_continuous = sorted(
        continuous_labels.keys()
    )

    active_discrete = sorted(
        discrete_labels.keys()
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
            modalities,

        "continuous_labels":
            continuous_labels,

        "discrete_labels":
            discrete_labels,

        "recognition_targets":
            recognition_targets,

        "forecast_targets":
            forecast_targets,

        "metadata":
            metadata,
    }