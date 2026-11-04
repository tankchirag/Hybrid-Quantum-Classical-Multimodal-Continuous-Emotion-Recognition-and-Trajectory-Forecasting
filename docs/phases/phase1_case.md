# Phase 1: CASE Detailed Pipeline Specification

> Implementation Status Note
>
> This document describes the original Phase 1 CASE design and
> intended implementation plan.
>
> During actual development several engineering adaptations were
> required to align the CASE dataset with the framework's universal
> training pipeline, history subsystem, forecasting subsystem,
> configuration system, and quantum model implementations.
>
> Therefore this document should be interpreted as the original
> architectural specification rather than the final implementation
> record.
>
> The authoritative record of what was actually implemented,
> validated, benchmarked and deployed is:
>
> docs/phase1_case_completion_report.md
>
> Any discrepancies between this specification and implementation
> should be resolved in favor of the completion report.

## Phase 1 — Visceral Domain Pipeline (CASE)

```text
  [CASE Dataset: Figshare Source]
  ├── interpolated/physiological/sub_XX.csv (1000 Hz, 8 Channels: ECG, BVP, GSR, RSP, SKT, 3x EMG)
  └── interpolated/annotations/sub_XX.csv   (20 Hz Joystick, [0.5, 9.5])
                     │
                     ▼
  [STAGE 1A: Pre-flight Audit & Inspection]
  Verify subject completeness (1-30), channel non-flatlining, and valid video IDs (1-8 vs 10-12).
                     │
                     ▼
  [STAGE 1B: Offline Preprocessing & Synchronization (@ 2 Hz Standard)]
- Filter non-stimulus video IDs (10, 11, 12)

- Session Z-score normalization per channel on valid trials

- Rescale joystick: [0.5, 9.5] -> [-1.0, 1.0]

- Decimate annotations: 20 Hz -> 2 Hz (10:1 boxcar filter)

- Window raw physiology: exactly 500 samples per 2 Hz step (T=10 -> shape (10, 500))

- Slice sliding windows: T=10 (5.0s), K=10 (5.0s), H=10 (5.0s), stride=1 (0.5s)

                     │
                     ▼
  [STAGE 1C: Anti-Leakage Subject Manifest Generation]
  Split YAML: Train (Subs 1-20), Val (Subs 21-25), Test (Subs 26-30)
  Emits: case_train_manifest.csv, case_val_manifest.csv, case_test_manifest.csv
                     │
                     ▼
  [STAGE 1D: Disk Dataset Loader (Universal Invariant Contract)]
  Emits standard batch dictionary: {"modalities": {8 streams of (B, T, 500)}, "history": (B, K, 2), ...}
                     │
                     ▼
  [STAGE 1E: 1D-CNN Waveform Encoders]
  8 Independent Conv1D encoders map (B, T, 500) -> (B, T, 128)
  Stacked representation: (B, T, 8, 128)
                     │
                     ▼
  [STAGE 1F: Universal Capacity-Matched Fusion]
  Select: B1 (Classical Attn) | B2 (Concat) | B3 (PQC Pooling) | QSAL (Quantum Attn)
  Decoupled register: Q in {4, 6, 8}
  Output Contract: Z_fused in R^(B x T x 256)
                     │
                     ▼
  [STAGE 1G: History-Aware State Conditioning Subsystem]
  Dynamic Adapter (Nc=2, Nd=0) -> 2-layer Causal GRU (H = 256 // 4 = 64-D) -> Post-fusion injection -> (B, T, 256)
                     │
                     ▼
  [STAGE 1H: Temporal Backbone & Forecasting Heads]
  Transformer Encoder (T=10) -> Transformer Decoder (H=10) -> Recognition Head + Forecasting Head
                     │
                     ▼
  [STAGE 1I: Benchmark Matrix & Metric Extraction]
  Evaluate CCC and RMSE at H in {4, 6, 10} for Thesis Chapter 5.1
```

---

## Step-by-Step Technical Blueprint: Alternatives, Pros, Cons & Recommendations
### Stage 1A: Pre-flight Audit & Inspection
#### Objectives
Before running any preprocessing script, an automated audit tool scans interpolated/physiological/ and interpolated/annotations/ across all 30 subjects.
#### Alternatives Considered
- Alternative 1 (Blind Preprocessing without Audit):
  - Method: Run preprocessing directly on raw files.
  - Pros: Saves 15 minutes of script setup.
  - Cons: High risk. Corrupted rows, dropped channels, or misaligned headers cause preprocessing to crash halfway or silently inject NaN tensors into model training.
- Alternative 2 (Comprehensive Pre-flight Diagnostic Audit):
  - Method: A standalone audit script (tools/inspect_case_dataset.py) verifies:
    1. All 30 subject files exist in both physiological/ and annotations/.
    2. Column headers match exactly: daqtime, ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap, video.
    3. No zero-variance flatlines (disconnected sensors) or unexpected NaNs exist.
    4. Label ranges stay bounded within $[0.5, 9.5]$.
  - Pros: Recommended. Produces reports/case_statistics.json and docs/datasets/case_dataset_spec.md as empirical artifacts for your thesis appendix.
  - Cons: Requires writing a dedicated verification script.
Decision: Alternative 2 (Comprehensive Audit).
### Stage 1B: Offline Preprocessing & Synchronization
#### Objectives
Convert raw, variable-length subject CSV files into uniform, rapidly loadable .npz slices on disk at the $2\text{ Hz}$ temporal standard.
#### Core Transformations
1. Stimulus Segment Extraction: Retain only emotional videos (IDs 1 through 8). Remove IDs 10, 11, and 12 (calibration screens and blue pauses).
2. Subject-Level Z-Score Standardization: Calculate $\mu$ and $\sigma$ for each of the 8 channels strictly across valid emotional clips per subject to remove baseline tonic drift:
3. $$x_{\text{norm}}^{(s, c)}(t) = \frac{x^{(s, c)}(t) - \mu^{(s, c)}}{\sigma^{(s, c)} + 10^{-6}}$$
4. Label Calibration & Decimation:
  - Rescale from $[0.5, 9.5]$ to $[-1.0, 1.0]$: $y_{\text{scaled}} = (y_{\text{raw}} - 5.0) / 4.5$.
  - Decimate from $20\text{ Hz}$ to $2\text{ Hz}$ using an anti-aliasing moving-average boxcar filter ($N = 10$):
  - $$y_{2\text{Hz}}[k] = \frac{1}{10} \sum_{j=0}^9 y_{20\text{Hz}}[10k + j]$$
5. Physiological Waveform Preservation: Each $500\text{ ms}$ frame ($2\text{ Hz}$) spans $1000\text{ Hz} \times 0.5\text{ s} = 500\text{ raw points}$. Store each channel window as an array of shape (10, 500).
#### Alternatives Considered
- Alternative 1 (Scalar Point-Averaging, $500 \to 1$):
  - Pros: Small file sizes; simple linear layers.
  - Cons: Rejected. Destroys morphological information: QRS peaks, heart rate variability, skin conductance response (SCR) rise slopes, and EMG energy bursts.
- Alternative 2 (Handcrafted Feature Extraction via Pan-Tompkins / TEAP):
  - Pros: Biologically interpretable summary metrics.
  - Cons: Rejected as primary input. Heuristic peak detectors fail during motion artifacts and block end-to-end gradient backpropagation through raw biosignals.
- Alternative 3 (Preserved 500-Sample Raw Waveform Frames):
  - Pros: Recommended. Preserves full physiological morphology; allows 1D-CNN encoders to learn spatial-temporal representations directly from physical waveforms.
  - Cons: Processed data footprint is $\approx 2.5\text{ GB}$ (negligible on modern storage).
Decision: Alternative 3 (Preserved Waveform Slices).
### Stage 1C: Anti-Leakage Split & Manifest Generation
#### Objectives
Partition data into disjoint training, validation, and test sets, and compile static manifest CSV indexes to eliminate runtime disk globbing.
#### Alternatives Considered
- Alternative 1 (Random Window Shuffling):
  - Method: Pool all slices from all subjects and split $70\% / 15\% / 15\%$ at random.
  - Cons: Fatal flaw. Because sliding windows advance by $\text{stride} = 1$ ($500\text{ ms}$), adjacent windows share $90\%$ identical sensory data. If window $i$ is in training and window $i+1$ from the same subject is in testing, catastrophic data leakage occurs, yielding artificially inflated, unpublishable correlation scores ($\text{CCC} > 0.95$).
- Alternative 2 (Subject-Independent Partitioning via YAML):
  - Method: Segregate subjects completely using data/splits/case_subject_split.yaml:
    - Train Split: Subjects 1 through 20 (20 participants, $\approx 67\%$).
    - Validation Split: Subjects 21 through 25 (5 participants, $\approx 16.5\%$).
    - Test Split: Subjects 26 through 30 (5 participants, $\approx 16.5\%$).
  - Pros: Recommended. Guaranteed zero subject leakage; fully reproducible and audit-compliant.
Decision: Alternative 2 (Subject-Independent Manifests).
### Stage 1D: Disk Dataset Loader (Universal Invariant Contract)
#### Objectives
Create datasets/loaders/case_disk_dataset.py to read manifest CSVs and emit batches matching the framework's universal interface.
```python
# Universal Output Contract emitted by CaseDiskDataset
batch = {
    "modalities": {
        "ecg": torch.Tensor,       # (B, T=10, 500)
        "bvp": torch.Tensor,       # (B, T=10, 500)
        "eda": torch.Tensor,       # (B, T=10, 500)
        "resp": torch.Tensor,      # (B, T=10, 500)
        "skt": torch.Tensor,       # (B, T=10, 500)
        "emg_zygo": torch.Tensor,  # (B, T=10, 500)
        "emg_coru": torch.Tensor,  # (B, T=10, 500)
        "emg_trap": torch.Tensor,  # (B, T=10, 500)
    },
    "history_continuous": torch.Tensor,  # (B, K=10, 2) [Valence, Arousal]
    "history_discrete": torch.Tensor,    # (B, K=10, 0) [Empty for CASE]
    "recognition_targets": {
        "continuous": torch.Tensor       # (B, T=10, 2)
    },
    "forecasting_targets": {
        "continuous": torch.Tensor       # (B, H=10, 2)
    },
    "metadata": {
        "subject_id": int,
        "video_id": int
    }
}
```

- Downstream modules (Fusion, History Engine, Transformer, Task Heads) receive this standard contract, remaining completely decoupled from CASE-specific file structures.
### Stage 1E: 1D-CNN Physiological Feature Encoders
#### Objectives
Map each raw channel frame from shape $(B, T, 500)$ to a dense latent representation $(B, T, 128)$.
#### Alternatives Considered
- Alternative 1 (Direct Linear Projection, $500 \to 128$):
  - Cons: Treats raw samples as unstructured independent features, ignoring temporal translation invariance and waveform shape.
- Alternative 2 (Recurrent Encoder / LSTM over 500 Points):
  - Cons: Recurrent step-by-step unrolling over 500 samples at every frame creates massive computational bottlenecks and slows training by $20\times$.
- Alternative 3 (Hierarchical 1D-CNN Frame Encoder):
  - Architecture:
    - Reshapes $(B, T, 500) \to (B \cdot T, 1, 500)$.
    - 3 Conv1D blocks with kernel sizes $[7, 5, 3]$, strides $[2, 2, 1]$, batch normalization, GELU activations, and max-pooling ($500 \to 250 \to 125 \to 31$).
    - Adaptive average pooling to 1 and linear projection to $128$-D.
    - Reshapes back to $(B, T, 128)$.
  - Pros: Recommended. Efficient, parameter-friendly, and captures localized waveform slopes and peaks.
Decision: Alternative 3 (1D-CNN Frame Encoder).
### Stage 1F: Universal Fusion Operators (Capacity-Matched Benchmarking)
#### Objectives
Stack the 8 physiological token streams into $(B, T, 8, 128)$ and fuse them into $\mathbf{Z}_{\text{fused}} \in \mathbb{R}^{B \times T \times 256}$.
```text
                     STAGE 1F: FUSION OPERATOR ALTERNATIVES
                     
     [B1: Classical Self-Attention]                 [B2: Unified Concatenation]
     Input: (B, T, 8, 128)                          Input: (B, T, 8, 128)
- Multi-Head Attention over 8 tokens           • Proj: 8 x Linear(128 -> 32)
- Linear(128 -> 256)                           • Channel Cat: 8 x 32 = 256
     Output: (B, T, 256)                            Output: (B, T, 256)
     
     [B3: Entangled PQC Pooling]                    [QSAL: Quantum Self-Attention]
     Input: (B, T, 8, 128)                          Input: (B, T, 8, 128)
- Compress: 8 x (128 -> 16)                    • Proj: 8 x (128 -> Q), Q in {4, 6, 8}
- MeanPool over 8 tokens -> (B, T, 16)         • Query PQC & Key PQC
- Proj: 16 -> 64 -> Q angles                   • 8 x 8 Quantum Attention Map
- Circular CNOT Entangled PQC                  • Attentive Token Pooling
- Linear(Q -> 256)                             • Linear(128 -> 256)
     Output: (B, T, 256)                            Output: (B, T, 256)
```
#### Decoupling Modalities from Qubits ($M \neq Q$)
- The 8 physiological channels are not mapped to 8 qubits.
- In B3, the 8 channels are mean-pooled down to $(B, T, 16)$ and encoded into $Q$ rotation angles.
- In QSAL, each of the 8 channel tokens is projected $128 \to Q$. Parameterized Query and Key circuits evaluate state overlaps across $Q$ qubits, generating an $8 \times 8$ cross-modal attention matrix across the 8 channels.
- Evaluating $Q \in \{4, 6, 8\}$ across both B3 and QSAL systematically tests the impact of quantum expressive capacity on biological signal fusion.
### Stage 1G: History-Aware State Conditioning Subsystem (Version A)
#### Objectives
Condition fused representations on historical emotion trajectories while preventing sensory bypass.
#### Protocol
1. Dynamic State Adapter: Projects continuous valence and arousal labels over the preceding $K = 10$ steps ($5.0\text{ seconds}$) into latent history space:
2. $$\mathbf{U}_{\text{hist}} = \text{GELU}(\text{LayerNorm}(\mathbf{W}_c \mathbf{Y}_{\text{hist\_cont}})) \in \mathbb{R}^{B \times 10 \times H}$$
3. Dynamic Capacity Bottleneck ($4:1$ Balanced Baseline):
4. $$H = \frac{d_{\text{transformer}}}{\text{ratio}} = \frac{256}{4} = 64\text{-D}$$
5. 
6. Allocating $64$-D to history alongside the $256$-D sensory stream ensures the model remains primarily driven by incoming physiology ($80\%$ sensory vs. $20\%$ history).
7. Trajectory Engine: A 2-layer causal GRU processes $\mathbf{U}_{\text{hist}}$ and outputs the final hidden state $\mathbf{h}_K \in \mathbb{R}^{B \times 64}$.
8. History Dropout ($p = 0.20$): During training, $\mathbf{h}_K$ is masked to $\mathbf{0}$ with $20\%$ probability, forcing the Transformer backbone to maintain strong reliance on the 8 physiological streams.
9. Post-Fusion Injection: Broadcast $\mathbf{h}_K$ across time $T$, concatenate with $\mathbf{Z}_{\text{fused}}$, and project back to $(B, T, 256)$:
10. $$\mathbf{Z}_{\text{in}} = \text{LayerNorm}\left(\mathbf{W}_{\text{inj}} [\mathbf{Z}_{\text{fused}} \,\Vert{}\, \mathbf{H}_{\text{broadcast}}]\right) \in \mathbb{R}^{B \times 10 \times 256}$$
11. 
12. Because injection occurs post-fusion, quantum circuit widths remain fixed, simulation speed is preserved, and sequence length $T$ is kept intact.
### Stage 1H: Temporal Backbone & Forecasting Heads
#### Objectives
Jointly estimate current emotional state and forecast multi-step future trajectories.
#### Architecture
- Transformer Encoder: 4 layers, 4 heads, $d_{\text{model}} = 256$ processing observation steps $T = 10$ ($5.0\text{ seconds}$) via causal self-attention. Emits memory tensor $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times 10 \times 256}$.
- Transformer Decoder: 4 layers, 4 heads, $d_{\text{model}} = 256$. A set of $H = 10$ learned positional query tokens $\mathbf{Q}_{\text{dec}} \in \mathbb{R}^{B \times 10 \times 256}$ attends over $\mathbf{H}_{\text{enc}}$ via cross-attention, decoding the future trajectory $\mathbf{H}_{\text{dec}} \in \mathbb{R}^{B \times 10 \times 256}$ in a single non-autoregressive forward pass.
- Task Heads:
  - Recognition Head: Linear projection $\mathbf{H}_{\text{enc}} \to \hat{\mathbf{Y}}_{\text{rec}} \in \mathbb{R}^{B \times 10 \times 2}$.
  - Forecasting Head: Linear projection $\mathbf{H}_{\text{dec}} \to \hat{\mathbf{Y}}_{\text{fore}} \in \mathbb{R}^{B \times 10 \times 2}$[cite: 30].
- Multi-Task Loss Objective:
- $$\mathcal{L}_{\text{total}} = 1.0 \cdot (1 - \text{CCC}(\mathbf{Y}_{\text{rec}}, \hat{\mathbf{Y}}_{\text{rec}})) + 1.0 \cdot \text{RMSE}(\mathbf{Y}_{\text{fore}}, \hat{\mathbf{Y}}_{\text{fore}})$$
### Stage 1I: Benchmark Matrix & Metric Extraction
#### Objectives
Run all baseline and proposed models across quantum capacities and history conditions to populate the primary results table for Thesis Chapter 5.1 (Visceral Physiological Affect Benchmark)[cite: 30].
#### Execution Order
1. Baseline B2 (Unified Concat): Establishes the non-attentive capacity control baseline[cite: 30].
2. Baseline B1 (Classical Attention): Establishes the parameter-matched classical self-attention baseline[cite: 30].
3. Baseline B3 (Entangled PQC): Evaluated across $Q \in \{4, 6, 8\}$ qubits[cite: 30].
4. Proposed QSAL (Quantum Self-Attention): Evaluated across $Q \in \{4, 6, 8\}$ qubits[cite: 30].
5. History Ablation Suite (on QSAL-4Q):
  - Condition A: history_encoder.enabled: false ($H = 0$, baseline)[cite: 30].
  - Condition B: ratio: 8 ($H = 32$, minimalist prior)[cite: 30].
  - Condition C: ratio: 4 ($H = 64$, standard baseline)[cite: 30].
  - Condition D: ratio: 2 ($H = 128$, high-capacity prior)[cite: 30].
6. Multi-Horizon Reporting: Every experiment evaluates forecasting accuracy at $H = 4$ ($2.0\text{ s}$), $H = 6$ ($3.0\text{ s}$), and $H = 10$ ($5.0\text{ s}$) to generate the trajectory degradation curves required for RQ3[cite: 29, 30].
#### Deliverable: Thesis Chapter 5.1 Results Table
Running this pipeline produces the following empirical benchmark table for your thesis report[cite: 30]:
| Model Index | Model Name | Fusion Mechanism | Qubits (Q) | History (H) | Recog. CCC (Valence) ↑ | Recog. CCC (Arousal) ↑ | Forecast CCC (H=4, 2s) | Forecast CCC (H=6, 3s) | Forecast CCC (H=10, 5s) | Overall RMSE ↓ |
|---|---|---|---:|---|---|---|---|---|---|---|
| M1 | Baseline B2 | Direct Concat | — | 4:1 (64) | eval | eval | eval | eval | eval | eval |
| M2 | Baseline B1 | Classical Attn | — | 4:1 (64) | eval | eval | eval | eval | eval | eval |
| M3 | Baseline B3 | PQC Pooling | 4 | 4:1 (64) | eval | eval | eval | eval | eval | eval |
| M4 | Baseline B3 | PQC Pooling | 6 | 4:1 (64) | eval | eval | eval | eval | eval | eval |
| M5 | Baseline B3 | PQC Pooling | 8 | 4:1 (64) | eval | eval | eval | eval | eval | eval |
| M6 | Proposed QSAL | Quantum Attn | 4 | 4:1 (64) | eval | eval | eval | eval | eval | eval |
| M7 | Proposed QSAL | Quantum Attn | 6 | 4:1 (64) | eval | eval | eval | eval | eval | eval |
| M8 | Proposed QSAL | Quantum Attn | 8 | 4:1 (64) | eval | eval | eval | eval | eval | eval |
| M9 | QSAL Ablation | Quantum Attn | 4 | Off (0) | eval | eval | eval | eval | eval | eval |
| M10 | QSAL Ablation | Quantum Attn | 4 | 8:1 (32) | eval | eval | eval | eval | eval | eval |
| M11 | QSAL Ablation | Quantum Attn | 4 | 2:1 (128) | eval | eval | eval | eval | eval | eval |

# Phase 1 Implementation Deviations and Engineering Notes

The final implementation remained faithful to the
scientific objectives of Phase 1 while introducing
several engineering adaptations based on actual
dataset characteristics and framework requirements.

## Major Differences

### Temporal Configuration

Original Design

T = 10
K = 10
H = 10

Final CASE Integration

Observation Length = 120

History Length = 120

Forecast Horizon = 20

Reason:

The framework's final continuous forecasting setup
uses longer observation histories and a 10 second
future prediction horizon.

---

### Dataset Contracts

Original Design

Dedicated CASE-specific batch format.

Final Implementation

Unified framework contract shared across:

- CASE
- Synthetic TSST
- Future OMG integration

Reason:

Allows all architectures to train without
dataset-specific model code.

---

### History Subsystem

Original Design

Conceptual History-Aware State Conditioning.

Final Implementation

Operational pipeline:

HistoryAdapter
    ->
HistoryRepresentation
    ->
HistoryEncoder
    ->
HistoryInjection

validated for:

- B1
- B2
- B3
- QSAL

---

### Forecasting

Original Design

Single planning horizon description.

Final Implementation

Unified forecasting subsystem:

Recognition Head

Forecasting Head

Transformer Encoder-Decoder

shared by all architectures.

---

### Quantum Models

Original Design

Conceptual benchmark plan for B3 and QSAL.

Final Implementation

Working implementations:

- Baseline B3
- QSAL

validated through:

Forward Pass
Loss
Backward
Training

under CASE.

---

### Integration Outcome

Successfully validated:

✅ Baseline B1

✅ Baseline B2

✅ Baseline B3

✅ QSAL

using:

✅ CASE Dataset

✅ History Conditioning

✅ Forecasting

✅ End-to-End Training

See:

docs/phase1_case_completion_report.md

### Git Branch & Workspace Discipline
Following your project convention, keep the integration branch clean[cite: 30]:
1. Current working branch: synthetic-multimodal-framework (contains the verified baseline and synthetic TSST code)[cite: 30].
2. Create and switch to the Phase 1 branch[cite: 30].

```bash
git checkout synthetic-multimodal-framework
git pull
git checkout -b phase1-case-benchmark
git push -u origin phase1-case-benchmark
```

3. Implement in order:
  - Stage 1A: tools/inspect_case_dataset.py $\to$ reports/case_statistics.json[cite: 30]
  - Stage 1B: datasets/preprocessing/case_preprocessor.py $\to$ data/processed/case/*.npz[cite: 30]
  - Stage 1C: data/splits/case_subject_split.yaml and build_case_manifest.py $\to$ manifest CSVs[cite: 30]
  - Stage 1D & 1E: case_disk_dataset.py and case_physio_encoders.py[cite: 30]
  - Stage 1F to 1I: Integration check $\to$ Run B2, B1, B3, QSAL $\to$ Record Chapter 5.1 results[cite: 30].
