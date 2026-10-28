"""
models.encoders.encoder_registry
================================

Central encoder registry.

Purpose
-------
Provide a single location for resolving
modality names into encoder classes.

Supports:

    YAML-driven modality activation

    Modality ablation studies

    CASE

    OMG

    Future datasets
"""

from __future__ import annotations

from .ecg_encoder import ECGEncoder
from .eda_encoder import EDAEncoder
from .eeg_encoder import EEGEncoder
from .emg_encoder import EMGEncoder
from .resp_encoder import RESPEncoder
from .bpm_encoder import BPMEncoder

from .eye_encoder import EyeEncoder

from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder
from .text_encoder import TextEncoder


from .bvp_encoder import BVPEncoder
from .gsr_encoder import GSREncoder
from .rsp_encoder import RSPEncoder
from .skt_encoder import SKTEncoder

from .emg_zygo_encoder import (
    EMGZygoEncoder,
)

from .emg_coru_encoder import (
    EMGCoruEncoder,
)

from .emg_trap_encoder import (
    EMGTrapEncoder,
)


ENCODER_REGISTRY = {

    # ----------------------------------
    # Physiological
    # ----------------------------------

    "ecg": ECGEncoder,
    "eda": EDAEncoder,
    "eeg": EEGEncoder,
    "emg": EMGEncoder,
    "resp": RESPEncoder,
    "bpm": BPMEncoder,
    "bvp": BVPEncoder,
    "gsr": GSREncoder,
    "rsp": RSPEncoder,
    "skt": SKTEncoder,

    "emg_zygo": EMGZygoEncoder,
    "emg_coru": EMGCoruEncoder,
    "emg_trap": EMGTrapEncoder,


    # ----------------------------------
    # Behavioral
    # ----------------------------------

    "eye": EyeEncoder,

    "audio": AudioEncoder,
    "video": VideoEncoder,
    "text": TextEncoder,
}


def has_encoder(
    modality_name: str,
) -> bool:

    return (
        modality_name
        in ENCODER_REGISTRY
    )


def get_encoder_class(
    modality_name: str,
):

    return (
        ENCODER_REGISTRY.get(
            modality_name
        )
    )