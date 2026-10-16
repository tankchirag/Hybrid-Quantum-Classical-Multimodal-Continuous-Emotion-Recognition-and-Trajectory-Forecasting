"""
utils.history_validation
========================

History Encoder
Configuration Validation.
"""


def validate_history_config(config):
    """
    Validate history encoder settings.

    Raises
    ------
    ValueError
        For invalid configurations.
    """

    history_cfg = config.history_encoder

    if not history_cfg.enabled:
        return

    ratio = getattr(
        history_cfg,
        "ratio",
        None,
    )

    auto_scale = getattr(
        history_cfg,
        "auto_scale",
        False,
    )

    if auto_scale:

        if ratio is None:
            raise ValueError(
                "history_encoder.auto_scale=True "
                "requires history_encoder.ratio."
            )

        if ratio <= 0:
            raise ValueError(
                "history_encoder.ratio "
                "must be > 0."
            )

        transformer_dim = (
            config.model.transformer_dim
        )

        if transformer_dim % ratio != 0:
            raise ValueError(
                f"transformer_dim="
                f"{transformer_dim} "
                f"must be divisible by "
                f"ratio={ratio}."
            )

    history_dropout = getattr(
        history_cfg,
        "history_dropout",
        0.0,
    )

    if not (
        0.0 <= history_dropout <= 1.0
    ):
        raise ValueError(
            "history_dropout "
            "must be within [0,1]."
        )