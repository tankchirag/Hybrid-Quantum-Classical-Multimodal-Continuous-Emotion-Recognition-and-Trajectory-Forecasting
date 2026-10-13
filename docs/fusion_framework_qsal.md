# Quantum Self-Attention Fusion Framework

## QSAL (Proposed Method)

---

# 1. Purpose

The Quantum Self-Attention Fusion Framework implements the proposed QSAL architecture.

QSAL combines:

- Quantum Representation Learning
- Classical Attention
- Multimodal Fusion

to create a hybrid quantum-classical fusion mechanism.

---

# 2. Position in Architecture

Raw Signals
    ↓
Encoders
    ↓
Tokenizer
    ↓
QSAL Fusion
    ↓
Transformer
    ↓
Recognition Head
    ↓
Forecasting Head

---

# 3. Scientific Motivation

Classical self-attention learns Query, Key, and Value projections using linear transformations.

QSAL replaces these projections with parameterized quantum circuits.

Traditional Attention

Q = Wx

K = Wx

V = Wx

QSAL

Q = PQC(x)

K = PQC(x)

V = PQC(x)

This enables quantum-enhanced attention representations.

---

# 4. Design Goals

✅ Dataset Agnostic

✅ Label Agnostic

✅ Continuous Labels

✅ Discrete Labels

✅ Recognition Compatible

✅ Forecasting Compatible

✅ Variable Source Count

✅ Variable Qubit Counts

✅ Variable Circuit Depth

✅ Config Driven

✅ Future Proof

---

# 5. Input Representation

Input:

{
    source_name:
        (B,T,128)
}

Stacked into:

(B,T,N,128)

where:

N = Number of Active Sources

---

# 6. Quantum Input Projection

Each token is projected:

128

↓

Q

using:

Linear(128 → Q)

where:

Q ∈ {4,6,8}

The projected values form the quantum input state.

The quantum dimension is configurable through the
experiment configuration file.

---

# 7. Quantum Query Circuit

The first PQC generates Query representations.

Input:

(B,T,N,Q)

↓

Q-PQC

↓

(B,T,N,Q)

where:

Q ∈ {4,6,8}

---

# 8. Quantum Key Circuit

The second PQC generates Key representations.

Input:

(B,T,N,Q)

↓

K-PQC

↓

(B,T,N,Q)

where:

Q ∈ {4,6,8}

---

# 9. Quantum Value Circuit

The third PQC generates Value representations.

Input:

(B,T,N,Q)

↓

V-PQC

↓

(B,T,N,Q)

where:

Q ∈ {4,6,8}

---

# 10. Gaussian Attention

Attention is computed using Gaussian similarity.

Inputs:

Q

K

V

↓

Gaussian Attention

↓

(B,T,N,4)

Purpose:

- learn source importance
- learn source interactions
- perform adaptive weighting

The Q-PQC, K-PQC, and V-PQC modules all share
the validated UnifiedPQC architecture.

Only their trainable parameters differ.

---

# 11. Source Pooling

Attention outputs are pooled across source dimension.

(B,T,N,Q)

↓

Mean Pooling

↓

(B,T,Q)

where:

Q ∈ {4,6,8}

The pooled vector becomes the fused quantum representation.

---

# 12. Transformer Projection

The quantum representation is projected:

Q

↓

fused_dim

↓

256

using:

Linear(Q → fused_dim)

Linear(fused_dim → 256)

where:

Q ∈ {4,6,8}

Output:

(B,T,256)

---

# 13. Why QSAL Is Different

B1

Classical self-Attention

↓

(B,T,256)

B2

Projection + Concatenation Fusion

↓

(B,T,256)

B3

Pure Quantum Fusion

↓

(B,T,256)

QSAL
 
Quantum Query
Quantum Key
Quantum Value
 
+
 
Gaussian Attention

↓

(B,T,256)

---

# 14. Recognition Compatibility

Supported Continuous Targets

- Valence
- Arousal
- Stress
- Workload
- Fatigue
- Dominance
- Engagement

Supported Discrete Targets

- Emotion Class
- Stress Class
- Workload Class
- Alertness Class
- Drowsiness Class
- KSS Score

---

# 15. Forecasting Compatibility

Past Signals

↓

QSAL Fusion

↓

Transformer

↓

Future Predictions

Examples:

- Future Valence
- Future Arousal
- Future Stress
- Future Workload

---

# 16. Computational Flow

Input

(B,T,N,128)

↓

Projection

(B,T,N,Q)

↓

Q-PQC

K-PQC

V-PQC

↓

Gaussian Attention

↓

(B,T,N,Q)

↓

Pooling

(B,T,Q)

↓

Projection

(B,T,fused_dim)

↓

Projection

(B,T,256)

where:

Q = Number of Qubits

N = Number of Active Sources

---

# 17. Validation

The QSAL framework was validated through:

- Configuration Validation
- Quantum Circuit Validation
- Gaussian Attention Validation
- Forward Pass Validation
- Smoke Testing
- Training Validation

Validated Configurations:

- 4Q D3
- 6Q D2
- 8Q D4

Validation confirmed:

✅ Correct Qubit Propagation

✅ Correct Depth Propagation

✅ Correct Q-PQC Construction

✅ Correct K-PQC Construction

✅ Correct V-PQC Construction

✅ Correct Gaussian Attention Execution

✅ Correct Transformer Compatibility

✅ Successful End-to-End Training

---

# 18. Summary

QSAL represents the proposed hybrid quantum-classical fusion method.

Input:

(B,T,N,128)

↓

Quantum Self-Attention

↓

(B,T,256)

The framework is:

✅ Dataset Agnostic

✅ Label Agnostic

✅ Continuous Labels

✅ Discrete Labels

✅ Recognition Compatible

✅ Forecasting Compatible

✅ Variable Source Count

✅ Future Proof

✅ Variable Qubit Counts

✅ Variable Circuit Depth

✅ Config Driven

✅ Transformer Compatible