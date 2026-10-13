"""
tests/trace/theory_registry.py

Central registry containing the theoretical architecture
specification for all audited models.

Purpose
-------
Acts as the single source of truth for:

1. Theory
2. Expected tensor transformations
3. Expected architectural components
4. Expected computational flow

Architecture audit scripts compare:

THEORY
vs
IMPLEMENTATION
vs
RUNTIME

to verify thesis consistency.
"""

from __future__ import annotations

# ==========================================================
# QSAL
# ==========================================================

QSAL_THEORY = {
    "name": "Quantum Self-Attention Fusion Framework",

    "architecture_flow": [
        "Raw Modalities",
        "Encoders",
        "Multimodal Feature Extractor",
        "Source Stack",
        "Quantum Projection",
        "Q-PQC",
        "K-PQC",
        "V-PQC",
        "Gaussian Attention",
        "Source Pooling",
        "Transformer Projection",
        "Transformer",
        "Recognition Head",
        "Forecasting Head",
    ],

    "required_modules": [
        "extractors",
        "fusion",
        "q_pqc",
        "k_pqc",
        "v_pqc",
        "attention",
        "temporal_model",
        "recognition_head",
        "forecasting_head",
    ],

    "stages": {

        "Raw Modalities": {
            "purpose":
                "Dataset-provided synchronized modality tensors.",
        },

        "Encoders": {
            "input": "(B,T,C)",
            "output": "(B,T,128)",
            "purpose":
                "Convert raw modalities into common latent embeddings.",
        },

        "Multimodal Feature Extractor": {
            "output": "dict[str,(B,T,128)]",
            "purpose":
                "Generate modality-specific embeddings.",
        },

        "Source Stack": {
            "input":
                "dict[str,(B,T,128)]",
            "output":
                "(B,T,N,128)",
            "purpose":
                "Stack active modality embeddings.",
        },

        "Quantum Projection": {
            "input":
                "(B,T,N,128)",
            "output":
                "(B,T,N,Q)",
            "purpose":
                "Project latent embeddings into quantum space.",
        },

        "Q-PQC": {
            "input":
                "(B,T,N,Q)",
            "output":
                "(B,T,N,Q)",
            "purpose":
                "Generate quantum query representations.",
        },

        "K-PQC": {
            "input":
                "(B,T,N,Q)",
            "output":
                "(B,T,N,Q)",
            "purpose":
                "Generate quantum key representations.",
        },

        "V-PQC": {
            "input":
                "(B,T,N,Q)",
            "output":
                "(B,T,N,Q)",
            "purpose":
                "Generate quantum value representations.",
        },

        "Gaussian Attention": {
            "input":
                "(B,T,N,Q)",
            "output":
                "(B,T,N,Q)",
            "purpose":
                "Compute source interactions and attention weighting.",
        },

        "Source Pooling": {
            "input":
                "(B,T,N,Q)",
            "output":
                "(B,T,Q)",
            "purpose":
                "Aggregate sources into a fused representation.",
        },

        "Transformer Projection": {
            "input":
                "(B,T,Q)",
            "output":
                "(B,T,256)",
            "purpose":
                "Prepare fused representation for transformer.",
        },

        "Transformer": {
            "encoder_output":
                "(B,T,256)",
            "decoder_output":
                "(B,H,256)",
            "purpose":
                "Temporal sequence modeling.",
        },

        "Recognition Head": {
            "purpose":
                "Recognition targets prediction.",
        },

        "Forecasting Head": {
            "purpose":
                "Future target prediction.",
        },
    },
}

# ==========================================================
# BASELINE B1
# ==========================================================

B1_THEORY = {
    "name": "Classical Self-Attention Baseline",

    "required_modules": [
        "extractors",
        "fusion",
        "attention",
        "temporal_model",
        "recognition_head",
        "forecasting_head",
    ],

    "architecture_flow": [
        "Raw Modalities",
        "Encoders",
        "Classical Self-Attention",
        "Transformer",
        "Recognition Head",
        "Forecasting Head",
    ],
}

# ==========================================================
# BASELINE B2
# ==========================================================

B2_THEORY = {
    "name": "Projection + Concatenation Baseline",

    "required_modules": [
        "extractors",
        "fusion",
        "temporal_model",
        "recognition_head",
        "forecasting_head",
    ],

    "architecture_flow": [
        "Raw Modalities",
        "Encoders",
        "Concatenation",
        "Projection",
        "Transformer",
        "Recognition Head",
        "Forecasting Head",
    ],
}

# ==========================================================
# BASELINE B3
# ==========================================================

B3_THEORY = {
    "name": "UnifiedPQC Quantum Fusion",

    "required_modules": [
        "extractors",
        "fusion",
        "unified_pqc",
        "temporal_model",
        "recognition_head",
        "forecasting_head",
    ],

    "architecture_flow": [
        "Raw Modalities",
        "Encoders",
        "Source Stack",
        "Quantum Projection",
        "UnifiedPQC",
        "Pooling",
        "Transformer",
        "Recognition Head",
        "Forecasting Head",
    ],
}

# ==========================================================
# REGISTRY
# ==========================================================

THEORY_REGISTRY = {
    "qsal": QSAL_THEORY,
    "b1": B1_THEORY,
    "b2": B2_THEORY,
    "b3": B3_THEORY,
}