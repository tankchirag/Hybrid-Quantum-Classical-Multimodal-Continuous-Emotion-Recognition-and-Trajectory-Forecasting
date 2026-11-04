def resolve_history_dim(config):

    if not config.history_encoder.enabled:
        return 0

    if config.history_encoder.auto_scale:

        fusion_dim = (
            config.model.transformer_dim
        )

        return (
            fusion_dim //
            config.history_encoder.ratio
        )

    if config.history_encoder.hidden_dim:
        return int(
            config.history_encoder.hidden_dim
        )

    return 0