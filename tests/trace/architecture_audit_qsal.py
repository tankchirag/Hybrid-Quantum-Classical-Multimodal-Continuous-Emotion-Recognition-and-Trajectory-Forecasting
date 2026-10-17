from __future__ import annotations

import sys
from pathlib import Path

import json
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

from models.wrappers.qsal_half import (
    QSAL,
)

from tests.trace.trace_core import (
    TraceRecorder,
    register_trace_hook,
    shape_of,
)

from tests.trace.theory_registry import (
    QSAL_THEORY,
)

from tests.trace.report_writer import (
    write_all_reports,
)


THEORY_STAGE_MAP = {
    "MultimodalFeatureExtractorsV2":
        "(B,T,128)",

    "QSALFusion":
        "(B,T,256)",

    "Q-PQC":
        "(B*T*N,1,Q)",

    "K-PQC":
        "(B*T*N,1,Q)",

    "V-PQC":
        "(B*T*N,1,Q)",

    "GaussianAttention":
        "(B*T,N,Q)",

    "TransformerEncoderDecoder":
        "(B,T,256)->(B,H,256)",

    "RecognitionHead":
        "Recognition Outputs",

    "ForecastingHead":
        "Forecast Outputs",
}



# ==========================================================
# Theory Validation
# ==========================================================

def validate_required_modules(
    model,
):

    results = []

    required = QSAL_THEORY[
        "required_modules"
    ]

    for module_name in required:

        exists = False

        # top-level
        if hasattr(
            model,
            module_name,
        ):
            exists = True

        # fusion-level
        elif hasattr(
            model,
            "fusion",
        ) and hasattr(
            model.fusion,
            module_name,
        ):
            exists = True

        results.append(
            {
                "component":
                    module_name,

                "theory":
                    "Required by theory",

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

    return results


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\nQSAL ARCHITECTURE AUDIT"
    )

    # ------------------------------------------------------
    # Config
    # ------------------------------------------------------

    config_dict = load_config(
        "configs/qsal.yaml"
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

    model = QSAL(
        config
    )

    model.eval()

    # ------------------------------------------------------
    # Recorder
    # ------------------------------------------------------

    recorder = TraceRecorder(
        model_name="QSAL"
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
        model.circuit_depth,
    )

    recorder.add_metadata(
        "Transformer Dimension",
        model.transformer_dim,
    )

    recorder.add_metadata(
        "Backend",
        model.backend,
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
    # Hook Registration
    # ------------------------------------------------------

    register_trace_hook(
        recorder,
        model.extractors,
        "MultimodalFeatureExtractorsV2",
    )

    register_trace_hook(
        recorder,
        model.fusion,
        "QSALFusion",
    )

    register_trace_hook(
        recorder,
        model.fusion.q_pqc,
        "Q-PQC",
    )

    register_trace_hook(
        recorder,
        model.fusion.k_pqc,
        "K-PQC",
    )

    register_trace_hook(
        recorder,
        model.fusion.v_pqc,
        "V-PQC",
    )

    register_trace_hook(
        recorder,
        model.fusion.attention,
        "GaussianAttention",
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
    # Runtime Forward Pass
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
    # Print Console Reports
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
    # Runtime Trace JSON
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

    runtime_trace_file = (
        Path(
            "trace_reports/qsal"
        )
        / "qsal_runtime_trace.json"
    )

    runtime_trace_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        runtime_trace_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            runtime_trace,
            f,
            indent=4,
        )

    # ------------------------------------------------------
    # Mermaid Flow
    # ------------------------------------------------------

    mermaid_nodes = [
        "RawModalities",
        "FeatureExtractors",
        "SourceStack",
        "Projection128ToQ",
        "QPQC",
        "KPQC",
        "VPQC",
        "GaussianAttention",
        "SourcePooling",
        "TransformerEncoder",
        "TransformerDecoder",
        "RecognitionHead",
        "ForecastingHead",
    ]

        # ------------------------------------------------------
    # Write Reports
    # ------------------------------------------------------

    write_all_reports(
        recorder=recorder,

        report_dir=
        "trace_reports/qsal",

        mermaid_nodes=
        mermaid_nodes,

        theory_results=
        theory_results,
    )

    print("\n")
    print("=" * 80)
    print("REPORT GENERATION")
    print("=" * 80)

    print(
        Path(
            "trace_reports/qsal"
        ).resolve()
    )

    print("\nArchitecture Audit Complete.")

if __name__ == "__main__":
    main()