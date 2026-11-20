"""Use experimental backend YAMLs with the existing CASE benchmark."""

from training import benchmark_adapter as original

EXPERIMENT_CONFIGS = {
    "b3": "configs/experiments/case_b3_torch_backprop.yaml",
    "qsal": "configs/experiments/case_qsal_torch_backprop.yaml",
}


def build_benchmark(args, benchmark_config):
    model = str(benchmark_config.model).lower()

    if model not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Optimized backend not configured for {model!r}"
        )

    specification = original.MODEL_REGISTRY[model]
    original_path = specification["config_path"]

    try:
        specification["config_path"] = EXPERIMENT_CONFIGS[model]
        return original.build_benchmark(args, benchmark_config)
    finally:
        specification["config_path"] = original_path
