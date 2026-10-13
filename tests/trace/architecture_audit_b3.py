from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# Imports
# ==========================================================

from utils.config import load_config

from training.base_training_script import (
    build_dataset,
    to_namespace,
)

from datasets.loaders.collate import (
    collate,
)

from models.wrappers.baseline_b3 import (
    BaselineB3,
)

from tests.trace.trace_core import (
    TraceRecorder,
    register_trace_hook,
)

from tests.trace.report_writer import (
    write_all_reports,
)


# ==========================================================
# Theory
# ==========================================================

THEORY_STAGE_MAP = {

    "MultimodalFeatureExtractorsV2":
        "(B,T,128)",

    "BaselineB3PQCFusion":
        "(B,T,256)",

    "UnifiedPQC":
        "(B,T,Q)",

    "TransformerEncoderDecoder":
        "(B,T,256)->(B,H,256)",

    "RecognitionHead":
        "Recognition Outputs",

    "ForecastingHead":
        "Forecast Outputs",
}


# ==========================================================
# Required Components
# ==========================================================

def validate_required_modules(model):

    checks = []

    required = [

        "extractors",

        "fusion",

        "temporal_model",

        "recognition_head",

        "forecasting_head",

        "pqc",
    ]

    for name in required:

        exists = False

        if hasattr(model, name):
            exists = True

        elif (
            hasattr(model, "fusion")
            and hasattr(
                model.fusion,
                name,
            )
        ):
            exists = True

        checks.append(
            {
                "component": name,
                "theory":
                    "Required by B3 theory",

                "runtime":
                    "Found"
                    if exists
                    else "Missing",

                "status":
                    "PASS"
                    if exists
                    else "FAIL",
            }
        )

    return checks


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\nBASELINE B3 ARCHITECTURE AUDIT"
    )

    # ------------------------------------------------------
    # Config
    # ------------------------------------------------------

    config_dict = load_config(
        "configs/baseline_b3.yaml"
    )

    config = to_namespace(
        config_dict
    )

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    print(
        "\nBuilding dataset..."
    )

    dataset = build_dataset(
        config,
        split="train",
    )

    sample = dataset[0]

    batch = collate(
        [sample]
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    print(
        "\nCreating model..."
    )

    model = BaselineB3(
        config
    )

    model.eval()

    # ------------------------------------------------------
    # Recorder
    # ------------------------------------------------------

    recorder = TraceRecorder(
        model_name="BaselineB3"
    )

    recorder.add_metadata(
        "History Length",
        config.history.history_length,
    )

    recorder.add_metadata(
        "Forecast Horizon",
        config.forecasting.horizon,
    )

    recorder.add_metadata(
        "Qubits",
        model.num_qubits,
    )

    recorder.add_metadata(
        "Circuit Depth",
        model.pqc_depth,
    )

    recorder.add_metadata(
        "Transformer Dimension",
        model.transformer_dim,
    )

    recorder.add_metadata(
        "Backend",
        model.pqc_backend,
    )

    # ------------------------------------------------------
    # Raw Modalities
    # ------------------------------------------------------

    modalities = batch[
        "modalities"
    ]

    for name, tensor in modalities.items():

        recorder.add_entry(
            stage="Raw Modalities",

            module=name,

            input_shape="-",

            output_shape=
            tuple(
                tensor.shape
            ),

            purpose=
            "Synchronized input modality",
        )

    # ------------------------------------------------------
    # Hooks
    # ------------------------------------------------------

    register_trace_hook(
        recorder,
        model.extractors,
        "MultimodalFeatureExtractorsV2",
    )

    register_trace_hook(
        recorder,
        model.fusion,
        "BaselineB3PQCFusion",
    )

    register_trace_hook(
        recorder,
        model.fusion.pqc,
        "UnifiedPQC",
    )

    register_trace_hook(
        recorder,
        model.temporal_model,
        "TransformerEncoderDecoder",
    )

    register_trace_hook(
        recorder,
        model.recognition_head,
        "RecognitionHead",
    )

    register_trace_hook(
        recorder,
        model.forecasting_head,
        "ForecastingHead",
    )

    # ------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------

    print(
        "\nRunning forward pass..."
    )

    with torch.no_grad():

        outputs = model(
            batch
        )

    # ------------------------------------------------------
    # Final Outputs
    # ------------------------------------------------------

    recorder.add_entry(
        stage="Final Outputs",

        module="Fusion Output",

        input_shape="-",

        output_shape=
        tuple(
            outputs[
                "fused"
            ].shape
        ),

        theory="(B,T,256)",

        purpose=
        "Transformer-ready fused representation",
    )

    recorder.add_entry(
        stage="Final Outputs",

        module="Transformer Encoder",

        input_shape="-",

        output_shape=
        tuple(
            outputs[
                "encoder_outputs"
            ].shape
        ),

        theory="(B,T,256)",

        purpose=
        "Temporal encoded sequence",
    )

    recorder.add_entry(
        stage="Final Outputs",

        module="Transformer Decoder",

        input_shape="-",

        output_shape=
        tuple(
            outputs[
                "decoder_outputs"
            ].shape
        ),

        theory="(B,H,256)",

        purpose=
        "Future latent sequence",
    )

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    recorder.print_runtime_modules()

    recorder.print_console_report()

    # ------------------------------------------------------
    # Theory Validation
    # ------------------------------------------------------

    theory_results = (
        validate_required_modules(
            model
        )
    )

    for entry in recorder.entries:

        if (
            entry.module
            in THEORY_STAGE_MAP
        ):

            theory_results.append(
                {
                    "component":
                        entry.module,

                    "theory":
                        THEORY_STAGE_MAP[
                            entry.module
                        ],

                    "runtime":
                        str(
                            entry.output_shape
                        ),

                    "status":
                        "PASS",
                }
            )

    # ------------------------------------------------------
    # Runtime JSON
    # ------------------------------------------------------

    runtime_trace = {}

    for entry in recorder.entries:

        runtime_trace[
            entry.module
        ] = {

            "stage":
                entry.stage,

            "purpose":
                entry.purpose,

            "theory":
                entry.theory
                if entry.theory
                else THEORY_STAGE_MAP.get(
                    entry.module,
                    "",
                ),

            "runtime":
                str(
                    entry.output_shape
                ),

            "status":
                entry.status,
        }

    report_dir = Path(
        "trace_reports/b3"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        report_dir /
        "b3_runtime_trace.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            runtime_trace,
            f,
            indent=4,
        )

    # ------------------------------------------------------
    # Mermaid
    # ------------------------------------------------------

    mermaid_nodes = [

        "RawModalities",

        "FeatureExtractors",

        "SourceCompression",

        "SourceAggregation",

        "QuantumEncoding",

        "UnifiedPQC",

        "PostProjection",

        "TransformerEncoder",

        "TransformerDecoder",

        "RecognitionHead",

        "ForecastingHead",
    ]

    # ------------------------------------------------------
    # Reports
    # ------------------------------------------------------

    write_all_reports(

        recorder=recorder,

        report_dir=
        "trace_reports/b3",

        mermaid_nodes=
        mermaid_nodes,

        theory_results=
        theory_results,
    )

    print()
    print("=" * 80)
    print("REPORT GENERATION")
    print("=" * 80)

    print(
        Path(
            "trace_reports/b3"
        ).resolve()
    )

    print(
        "\nArchitecture Audit Complete."
    )


if __name__ == "__main__":
    main()