"""
ulm_tsst_synthetic.py
=====================

Unified Synthetic ULM-TSST Dataset

A research-grade synthetic multimodal time-series dataset
for:

    • Continuous Emotion Recognition

    • Continuous Emotion Forecasting

    • Stress Recognition

    • Workload Estimation

    • Fatigue Monitoring

    • Drowsiness Detection

    • Human State Forecasting

The dataset is:

    ✓ modality agnostic

    ✓ label agnostic

    ✓ dataset agnostic

    ✓ forecasting compatible

    ✓ configurable

    ✓ long-duration clip based

    ✓ native-frequency sensor based

    ✓ realistic multimodal simulation

Architecture
------------

Subject Model
      ↓

TSST Protocol
      ↓

Latent State
      ↓

Physiology
      ↓

Sensors

Outputs
-------

sample = {

    "modalities": {...},

    "continuous_labels": {...},

    "discrete_labels": {...},

    "masks": {...},

    "metadata": {...}
}
"""

from __future__ import annotations

import os
from typing import Dict
from typing import Any

import numpy as np
import torch

from datasets.base_dataset import (
    BaseMultimodalDataset
)

from datasets.generators.subject_model import (
    SubjectModelGenerator
)

from datasets.generators.tsst_protocol import (
    TSSTProtocolGenerator
)

from datasets.generators.latent_state_generator import (
    LatentStateGenerator
)

from datasets.generators.continuous_label_generator import (
    ContinuousLabelGenerator
)

from datasets.generators.discrete_label_generator import (
    DiscreteLabelGenerator
)

from datasets.generators.physiological_generator import (
    PhysiologicalGenerator
)

from datasets.generators.sensor_generator import (
    SensorGenerator
)


class SyntheticULMTSSTDataset(
    BaseMultimodalDataset
):
    """
    Synthetic multimodal dataset emulator.

    Parameters
    ----------
    num_samples : int

    duration_seconds : int

    modalities : dict

    continuous_labels : dict

    discrete_labels : dict

    sampling_rates : dict

    save_dir : str | None

    device : str
    """

    def __init__(
        self,
        num_samples: int,
        duration_seconds: int,
        modalities: Dict[str, bool],
        continuous_labels: Dict[str, bool],
        discrete_labels: Dict[str, bool],
        sampling_rates: Dict[str, int],
        label_sampling_rate: int = 25,
        num_subjects: int = 30,
        num_sessions: int = 5,
        save_dir: str | None = None,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:

        super().__init__(
            T=duration_seconds,
            S=0,
        )

        self.device = torch.device(
            device
        )

        self.num_samples = num_samples

        self.duration_seconds = (
            duration_seconds
        )

        self.label_fs = (
            label_sampling_rate
        )

        self.modalities = modalities

        self.continuous_cfg = (
            continuous_labels
        )

        self.discrete_cfg = (
            discrete_labels
        )

        self.sampling_rates = (
            sampling_rates
        )

        self.num_subjects = (
            num_subjects
        )

        self.num_sessions = (
            num_sessions
        )

        self.save_dir = save_dir

        self.seed = seed

        if save_dir is not None:
            os.makedirs(
                save_dir,
                exist_ok=True,
            )

        # ----------------------------------
        # Core generators
        # ----------------------------------

        self.subject_generator = (
            SubjectModelGenerator(
                seed=seed
            )
        )

        self.protocol_generator = (
            TSSTProtocolGenerator(
                duration_seconds=
                duration_seconds,
                sampling_rate=
                label_sampling_rate,
                device=device,
            )
        )

        self.latent_generator = (
            LatentStateGenerator(
                sampling_rate=
                label_sampling_rate,
                seed=seed,
                device=device,
            )
        )

        self.continuous_generator = (
            ContinuousLabelGenerator(
                continuous_labels
            )
        )

        self.discrete_generator = (
            DiscreteLabelGenerator(
                discrete_labels
            )
        )

        self.physiology_generator = (
            PhysiologicalGenerator(
                device=device,
            )
        )

        self.sensor_generator = (
            SensorGenerator(
                modality_config=
                modalities,

                duration_seconds=
                duration_seconds,

                label_sampling_rate=
                label_sampling_rate,

                sampling_rates=
                sampling_rates,

                seed=seed,

                device=device,
            )
        )

    # ==================================================
    # Metadata
    # ==================================================

    def _metadata(
        self,
        idx: int,
    ) -> Dict[str, Any]:

        subject_id = (
            idx %
            self.num_subjects
        )

        session_id = (
            idx //
            self.num_subjects
        ) % self.num_sessions

        return {

            "clip_id": idx,

            "subject_id":
                subject_id,

            "session_id":
                session_id,

            "duration_seconds":
                self.duration_seconds,
        }

    # ==================================================
    # Masks
    # ==================================================

    def _modality_masks(
        self,
        modalities:
        Dict[str, torch.Tensor],
    ):

        masks = {}

        for name, value in (
            modalities.items()
        ):

            masks[name] = torch.ones(
                value.shape[0],
                dtype=torch.bool,
            )

        return masks

    # ==================================================
    # Saving
    # ==================================================

    def _save_sample(
        self,
        sample,
        meta,
    ) -> None:

        if self.save_dir is None:
            return

        path = os.path.join(

            self.save_dir,

            f"clip_"
            f"{meta['clip_id']:06d}"
            ".npz"
        )

        np_dict = {}

        # ----------------------------------
        # Metadata
        # ----------------------------------

        np_dict["meta_clip_id"] = np.asarray(
            meta["clip_id"],
            dtype=np.int32,
        )

        np_dict["meta_subject_id"] = np.asarray(
            meta["subject_id"],
            dtype=np.int32,
        )

        np_dict["meta_session_id"] = np.asarray(
            meta["session_id"],
            dtype=np.int32,
        )

        np_dict[
            "meta_duration_seconds"
        ] = np.asarray(
            meta["duration_seconds"],
            dtype=np.int32,
        )

        # ----------------------------------
        # Modalities
        # ----------------------------------

        for name, tensor in (
            sample["modalities"]
            .items()
        ):

            np_dict[
                f"mod_{name}"
            ] = (
                tensor
                .cpu()
                .numpy()
            )

        # ----------------------------------
        # Continuous Labels
        # ----------------------------------

        for name, tensor in (
            sample[
                "continuous_labels"
            ].items()
        ):

            np_dict[
                f"cont_{name}"
            ] = (
                tensor
                .cpu()
                .numpy()
            )

        # ----------------------------------
        # Discrete Labels
        # ----------------------------------

        for name, tensor in (
            sample[
                "discrete_labels"
            ].items()
        ):

            np_dict[
                f"disc_{name}"
            ] = (
                tensor
                .cpu()
                .numpy()
            )

        np.savez_compressed(

            path,

            **np_dict,
        )


    # ==================================================
    # Length
    # ==================================================

    def __len__(
        self
    ) -> int:

        return (
            self.num_samples
        )

    # ==================================================
    # Main
    # ==================================================

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:

        metadata = self._metadata(
            idx
        )

        subject_profile = (
            self.subject_generator
            .generate(
                metadata[
                    "subject_id"
                ]
            )
        )

        protocol = (
            self.protocol_generator
            .generate()
        )

        latent = (
            self.latent_generator
            .generate(
                subject=subject_profile,
                protocol=protocol,
            )
        )

        continuous_labels = (
            self.continuous_generator
            .generate(
                latent
            )
        )

        discrete_labels = (
            self.discrete_generator
            .generate(
                latent
            )
        )

        physiology = (
            self.physiology_generator
            .generate(
                latent=latent,
                subject=
                subject_profile,
            )
        )

        sensor_output = (
            self.sensor_generator
            .generate(
                latent=latent,
                phys=physiology,
            )
        )

        sample = {

            "modalities":
                sensor_output.modalities,

            "continuous_labels":
                continuous_labels.labels,

            "discrete_labels":
                discrete_labels.labels,

            "masks":
                self._modality_masks(
                    sensor_output.modalities
                ),

            "metadata":
                metadata,
        }

        self._save_sample(
            sample,
            metadata,
        )

        return sample