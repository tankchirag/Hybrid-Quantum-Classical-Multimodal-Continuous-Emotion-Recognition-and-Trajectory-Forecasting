# Quantum Framework

## Hybrid Quantum-Classical Multimodal Continuous Emotion Recognition and Trajectory Forecasting

---

# 1. Purpose

The Quantum Framework introduces quantum representation learning into the proposed multimodal affective computing architecture.

Its purpose is to investigate whether Parameterized Quantum Circuits (PQCs) can improve:

- Multimodal Fusion
- Attention Generation
- Continuous Emotion Recognition
- Discrete Emotion Classification
- Future State Forecasting
- Temporal Trajectory Prediction

The framework provides a shared quantum foundation used by:

- Baseline B3
- QSAL

while remaining compatible with the common transformer-based learning pipeline.

---

# 2. Design Philosophy

Most multimodal affective computing systems rely entirely on classical fusion mechanisms such as:

- Concatenation
- Mean Pooling
- Cross Attention
- Self Attention
- Transformer Fusion

The proposed framework investigates whether quantum representation learning can provide an alternative information processing space for multimodal learning.

Core design principles:

✅ Hybrid Quantum-Classical Learning

✅ Shared Quantum Infrastructure

✅ Dataset Agnostic

✅ Modality Agnostic

✅ Configurable Quantum Capacity

✅ Transformer Compatible

✅ Recognition Compatible

✅ Forecasting Compatible

---

# 3. Position in Architecture

Complete Architecture

Raw Sensors
    ↓
Synchronizer
    ↓
Window Sampler
    ↓
Encoder Framework
    ↓
Tokenizer
    ↓
Quantum Framework
    ↓
Transformer
    ↓
Recognition Head
    ↓
Forecasting Head

The Quantum Framework acts as the interface between:

Multimodal Representations

and

Temporal Learning

---

# 4. Research Objective

The central research question is:

Can quantum representation learning improve multimodal fusion and attention generation compared with classical approaches?

To investigate this question, two quantum architectures were developed:

### Baseline B3

Parameterized Quantum Circuit Fusion

### QSAL

Quantum Self-Attention Learning

Both architectures share the same underlying quantum circuit.

---

# 5. Unified Parameterized Quantum Circuit

File:

quantum/unified_pqc.py

The UnifiedPQC serves as the common quantum representation learner used throughout the thesis.

A single validated circuit architecture is shared by:

- Baseline B3
- QSAL
- Future Quantum Extensions

This enables fair comparison across experiments.

---

# 6. UnifiedPQC Architecture

The circuit consists of five stages.

Input Features
        ↓
Angle Encoding
        ↓
Data Reuploading
        ↓
Trainable Rotations
        ↓
Ring Entanglement
        ↓
Pauli-Z Measurements

The design balances expressive power, scalability, and computational feasibility.

---

# 7. Angle Encoding

Classical features are converted into quantum states using rotational gates.

For every qubit:

RY(angle)

is applied.

Example:

4 Qubits

q0 ← RY(x0)

q1 ← RY(x1)

q2 ← RY(x2)

q3 ← RY(x3)

This maps classical information into the quantum state space.

---

# 8. Data Reuploading

The framework employs data reuploading to increase representational capacity.

Instead of encoding information only once, the original features are repeatedly reintroduced at every quantum layer.

Conceptually:

Encode
    ↓
Process
    ↓
Encode Again
    ↓
Process Again

This allows deeper quantum representations without increasing the number of qubits.

---

# 9. Trainable Quantum Layers

Each quantum layer contains trainable parameters.

For every qubit:

RZ(φ)

RY(θ)

are applied.

Trainable parameters:

θ ∈ R(depth × qubits)

φ ∈ R(depth × qubits)

These parameters are optimized through gradient-based learning.

---

# 10. Ring Entanglement

Following trainable rotations, neighboring qubits are entangled using a ring topology.

Example:

q0 → q1

q1 → q2

q2 → q3

q3 → q0

implemented using CNOT gates.

Advantages:

- Uniform Connectivity
- Scalability
- Stable Architecture
- Efficient Implementation

---

# 11. Measurement Layer

The final quantum state is transformed back into a classical representation using expectation values.

Measurement:

Pauli-Z Expectation

Output:

(Q)

where:

Q = Number of Qubits

Example:

4 Qubits

↓

4 Measured Values

---

# 12. Supported Configurations

The framework currently supports:

4 Qubits

6 Qubits

8 Qubits

and configurable circuit depths:

Depth 2

Depth 3

Depth 4

This enables systematic evaluation of different quantum capacities.

---

# 13. Quantum Capacity Scaling

Trainable parameters scale according to:

2 × Depth × Qubits

Examples:

| Qubits | Depth | Parameters |
|----------|----------|----------|
| 4 | 3 | 24 |
| 6 | 3 | 36 |
| 8 | 4 | 64 |

Increasing qubits and depth increases representational capacity while also increasing computational complexity.

---

# 14. Baseline B3 Integration

Baseline B3 evaluates pure quantum fusion.

Pipeline:

Multimodal Embeddings
        ↓
Source Compression
        ↓
Source Aggregation
        ↓
UnifiedPQC
        ↓
Transformer Projection
        ↓
Transformer

Scientific Objective:

Can a Parameterized Quantum Circuit replace classical fusion mechanisms?

---

# 15. QSAL Integration

QSAL extends the framework by introducing quantum self-attention.

Pipeline:

Multimodal Embeddings
         ↓
Q-PQC

K-PQC

V-PQC
         ↓
Gaussian Attention
         ↓
Transformer Projection
         ↓
Transformer

Three independent UnifiedPQC instances generate:

- Quantum Queries
- Quantum Keys
- Quantum Values

Scientific Objective:

Can quantum-generated attention representations improve multimodal fusion?

---

# 16. Validation

The Quantum Framework was validated through multiple stages.

Validated:

✅ 4-Qubit Configurations

✅ 6-Qubit Configurations

✅ 8-Qubit Configurations

✅ Variable Circuit Depths

✅ Circuit Construction

✅ Circuit Visualization

✅ Forward Execution

✅ Baseline B3 Training

✅ QSAL Training

✅ Transformer Compatibility

---

# 17. Advantages

The framework provides:

- Shared Quantum Backbone
- Configurable Qubit Counts
- Configurable Circuit Depth
- Transformer Compatibility
- Recognition Compatibility
- Forecasting Compatibility
- Dataset Agnostic Design
- Modality Agnostic Design
- Reusable Experimental Infrastructure

---

# 18. Summary

The Quantum Framework introduces a configurable quantum learning layer into the proposed Hybrid Quantum-Classical Multimodal Architecture.

The central component is the Unified Parameterized Quantum Circuit, which serves as the common quantum representation learner for both Baseline B3 and QSAL.

The framework enables systematic evaluation of quantum fusion and quantum attention mechanisms while preserving compatibility with the remaining classical learning pipeline.