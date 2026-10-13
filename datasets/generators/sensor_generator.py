"""
sensor_generator.py
===================

Multimodal Sensor Generator

Transforms physiological state variables into
realistic sensor observations.

Pipeline
--------

TSST
 ↓

Latent State
 ↓

Physiology
 ↓

Sensor Signals
 ↓

Synchronizer
 ↓

Encoders
 ↓

Models

Generated Modalities
--------------------

ECG

EDA

RESP

BPM

EEG

EMG

EYE

AUDIO

VIDEO

TEXT

Design Goals
------------

✓ Multimodal

✓ Realistic correlations

✓ Native sensor frequencies

✓ Configurable

✓ Dataset agnostic

✓ Modality agnostic

✓ Long recordings

✓ Forecasting compatible
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from datasets.generators.physiological_generator import (
    PhysiologicalStateOutput,
)

from datasets.generators.latent_state_generator import (
    LatentStateOutput,
)


# ==========================================================
# Output
# ==========================================================

@dataclass
class SensorOutput:
    """
    Generated modalities.
    """

    modalities: Dict[str, torch.Tensor]


# ==========================================================
# Generator
# ==========================================================

class SensorGenerator:
    """
    Generate sensor observations.

    Parameters
    ----------
    modality_config : dict

    sampling_rates : dict

    device : str
    """

    DEFAULT_FS = {

        "ecg": 1000,
        "eda": 128,
        "resp": 64,
        "bpm": 4,

        "eeg": 256,
        "emg": 1000,

        "eye": 120,

        "audio": 100,
        "video": 25,
        "text": 5,
    }

    def __init__(
        self,
        modality_config: Dict[str, bool],
        duration_seconds: int,
        label_sampling_rate: int = 25,
        sampling_rates: Dict[str, int] | None = None,
        seed: int = 42,
        device: str = "cpu",
    ):

        self.modalities = modality_config

        self.duration_seconds = duration_seconds

        self.label_fs = label_sampling_rate

        self.fs = (
            sampling_rates
            or self.DEFAULT_FS
        )

        self.device = torch.device(device)

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------
    # Utility
    # ------------------------------------------------------

    def _upsample(
        self,
        x: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:

        x = x.float()

        x = (
            torch.nn.functional
            .interpolate(
                x.view(1, 1, -1),
                size=target_length,
                mode="linear",
                align_corners=True,
            )
            .view(-1)
        )

        return x

    # ------------------------------------------------------
    # ECG
    # ------------------------------------------------------

    def _ecg(
        self,
        phys: PhysiologicalStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["ecg"]

        N = (
            self.duration_seconds
            * fs
        )

        hr = self._upsample(
            phys.heart_rate,
            N,
        )

        t = torch.linspace(
            0,
            self.duration_seconds,
            N,
            device=self.device,
        )

        freq = hr / 60.0

        ecg = (

            torch.sin(
                2 * np.pi * freq * t
            )

            +

            0.20
            *
            torch.sin(
                4 * np.pi * freq * t
            )

        )

        ecg += (

            0.02
            *
            torch.randn(
                N,
                device=self.device,
            )
        )

        return ecg.unsqueeze(-1)

    # ------------------------------------------------------
    # EDA
    # ------------------------------------------------------

    def _eda(
        self,
        phys: PhysiologicalStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["eda"]

        N = (
            self.duration_seconds
            * fs
        )

        sym = self._upsample(
            phys.sympathetic,
            N,
        )

        tonic = (

            0.2
            +
            1.5 * sym

        )

        phasic = (

            0.15
            *
            torch.abs(
                torch.randn(
                    N,
                    device=self.device,
                )
            )

        )

        eda = (

            tonic
            +
            phasic

        )

        return eda.unsqueeze(-1)

    # ------------------------------------------------------
    # RESP
    # ------------------------------------------------------

    def _resp(
        self,
        phys: PhysiologicalStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["resp"]

        N = (
            self.duration_seconds
            * fs
        )

        rr = self._upsample(
            phys.respiration_rate,
            N,
        )

        t = torch.linspace(
            0,
            self.duration_seconds,
            N,
        )

        resp = torch.sin(
            2
            * np.pi
            * (rr / 60)
            * t
        )

        return resp.unsqueeze(-1)

    # ------------------------------------------------------
    # BPM
    # ------------------------------------------------------

    def _bpm(
        self,
        phys: PhysiologicalStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["bpm"]

        N = (
            self.duration_seconds
            * fs
        )

        bpm = self._upsample(
            phys.heart_rate,
            N,
        )

        bpm += (

            0.5
            *
            torch.randn(
                N,
                device=self.device,
            )

        )

        return bpm.unsqueeze(-1)

    # ------------------------------------------------------
    # EEG
    # ------------------------------------------------------

    def _eeg(
        self,
        phys: PhysiologicalStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["eeg"]

        N = (
            self.duration_seconds
            * fs
        )

        cortical = self._upsample(
            phys.cortical_activation,
            N,
        )

        t = torch.linspace(
            0,
            self.duration_seconds,
            N,
            device=self.device,
        )

        alpha = torch.sin(
            2 * np.pi * 10 * t
        )

        beta = torch.sin(
            2 * np.pi * 20 * t
        )

        eeg = (

            alpha
            *
            (
                1.0
                - cortical
            )

            +

            beta
            *
            (
                1.0
                + 2.0 * cortical
            )

        )

        return eeg.unsqueeze(-1)

    # ------------------------------------------------------
    # EMG
    # ------------------------------------------------------

    def _emg(
        self,
        phys: PhysiologicalStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["emg"]

        N = (
            self.duration_seconds
            * fs
        )

        tension = self._upsample(
            phys.muscle_tension,
            N,
        )

        return (
            tension
            + 0.05 * torch.randn(N)
        ).unsqueeze(-1)

    # ------------------------------------------------------
    # Eye
    # ------------------------------------------------------

    def _eye(
        self,
        phys: PhysiologicalStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["eye"]

        N = (
            self.duration_seconds
            * fs
        )

        pupil = self._upsample(
            phys.pupil_size,
            N,
        )

        blink = (
            0.1 *
            torch.randn(N)
        )

        return torch.stack(
            [
                pupil,
                blink,
            ],
            dim=-1,
        )

    # ------------------------------------------------------
    # Audio
    # ------------------------------------------------------

    def _audio(
        self,
        latent: LatentStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["audio"]

        N = (
            self.duration_seconds
            * fs
        )

        arousal = self._upsample(
            latent.arousal,
            N,
        )

        stress = self._upsample(
            latent.stress,
            N,
        )

        pitch = (

            120
            +
            50 * arousal
            +
            20 * stress

        )

        energy = (

            0.5
            +
            0.8 * stress

        )

        return torch.stack(
            [
                pitch,
                energy,
            ],
            dim=-1,
        )

    # ------------------------------------------------------
    # Video
    # ------------------------------------------------------

    def _video(
        self,
        latent: LatentStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["video"]

        N = (
            self.duration_seconds
            * fs
        )

        valence = self._upsample(
            latent.valence,
            N,
        )

        arousal = self._upsample(
            latent.arousal,
            N,
        )

        facial_motion = arousal

        smile_intensity = (
            (valence + 1)
            / 2
        )

        return torch.stack(
            [
                facial_motion,
                smile_intensity,
            ],
            dim=-1,
        )

    # ------------------------------------------------------
    # Text
    # ------------------------------------------------------

    def _text(
        self,
        latent: LatentStateOutput,
    ) -> torch.Tensor:

        fs = self.fs["text"]

        N = (
            self.duration_seconds
            * fs
        )

        valence = self._upsample(
            latent.valence,
            N,
        )

        stress = self._upsample(
            latent.stress,
            N,
        )

        return torch.stack(
            [
                valence,
                stress,
            ],
            dim=-1,
        )

    # ------------------------------------------------------
    # Main
    # ------------------------------------------------------

    def generate(
        self,
        latent: LatentStateOutput,
        phys: PhysiologicalStateOutput,
    ) -> SensorOutput:

        modalities = {}

        if self.modalities.get("ecg", False):
            modalities["ecg"] = self._ecg(phys)

        if self.modalities.get("eda", False):
            modalities["eda"] = self._eda(phys)

        if self.modalities.get("resp", False):
            modalities["resp"] = self._resp(phys)

        if self.modalities.get("bpm", False):
            modalities["bpm"] = self._bpm(phys)

        if self.modalities.get("eeg", False):
            modalities["eeg"] = self._eeg(phys)

        if self.modalities.get("emg", False):
            modalities["emg"] = self._emg(phys)

        if self.modalities.get("eye", False):
            modalities["eye"] = self._eye(phys)

        if self.modalities.get("audio", False):
            modalities["audio"] = self._audio(latent)

        if self.modalities.get("video", False):
            modalities["video"] = self._video(latent)

        if self.modalities.get("text", False):
            modalities["text"] = self._text(latent)

        return SensorOutput(
            modalities=modalities
        )