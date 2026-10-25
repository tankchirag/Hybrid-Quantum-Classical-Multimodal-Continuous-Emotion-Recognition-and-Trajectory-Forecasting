# Phase 2: Expressive Affective Domain Benchmark (OMG-Emotion)
## Scientific Specification & Architectural Blueprint

**Author:** Tank Chirag Girishbhai (Student ID: 00147844)  
**Degree Programme:** M.Sc. Artificial Intelligence Engineering of Autonomous Systems  
**Institution:** Technische Hochschule Ingolstadt (THI), Faculty of Computer Science  
**Supervisors:** Prof. Dr. Ignacio Alvarez, Vanchha Chandrayan  
**Target Repository Location:** `docs/research/omg_expressive_pipeline.md`

---

## 1. Executive Summary & Design Scope

The objective of **Phase 2** is to benchmark the universal hybrid quantum–classical forecasting architecture on the **expressive behavioral domain** using the naturalistic One-Minute Gradual Emotion (OMG-Emotion) dataset [Barros et al., 2018].

While Phase 1 (CASE [Sharma et al., 2018]) benchmarks internal autonomic nervous system (ANS) visceral reactivity, Phase 2 evaluates outward socio-communicative expressions manifested across facial action dynamics, vocal prosody, and linguistic semantics.

### Research Questions Addressed in Phase 2

- **RQ1 (Expressive Cross-Modal Fusion):** Does the Quantum Self-Attention Layer (QSAL) model cross-modal interactions between vocal acoustics, facial kinematics, and semantic dialogue context more effectively than classical fusion baselines (B1, B2, B3)?
- **RQ2 (Behavioral Affect Recognition):** Can the framework estimate continuous Valence and Arousal at time \(t\) from conversational monologues?
- **RQ3 (Trajectory Forecasting across Dialogue Turns):** Does the integration of QSAL with the causal history adapter and Transformer decoder improve continuous trajectory forecasting (\(t+1 \dots t+H\)) across sequential communicative expressions?

### Phase 2 Expressive Domain Pipeline (OMG-Emotion)

```text
========================================================================================================================
                                     PHASE 2 EXPRESSIVE DOMAIN PIPELINE (OMG-EMOTION)
========================================================================================================================

  [Raw OMG-Emotion Dataset Source]
  ├── Continuous Dialogue Videos: 567 YouTube Monologues (~60s average, 25/30 fps)[cite: 22]
  ├── Segmented Utterances: 7,371 Turns (~3-8s duration, 16 kHz Audio, Transcripts)[cite: 22, 24, 27]
  └── Utterance Annotations: Evaluator Weighted Estimator (EWE) continuous (v, a) in [-1, 1] x [0, 1][cite: 22, 25, 26]
                     │
                     ▼
  [STAGE 2A: Pre-flight Audit & Dialogue Inspection (tools/inspect_omg_dataset.py)]
  ├── Audits monologue continuity, utterance start/end timestamps, text presence, and audio integrity[cite: 24]
  └── Emits: reports/omg_statistics.json & docs/datasets/omg_dataset_spec.md
                     │
                     ▼
  [STAGE 2B: Offline Preprocessing & Feature Caching (datasets/preprocessing/omg_preprocessor.py)]
  ├── Continuous Label Reconstruction: Monotonic PCHIP across utterance midpoints @ 2 Hz (dt = 500 ms)[cite: 29]
  ├── Offline Feature Extraction (Pretrained Encoders -> Pre-cached 128-D Embeddings):
  │   • Visual: Face Bounding Box -> ResNet18 (AffectNet pretrained) -> Temporal Pool over 500 ms -> (T, 128)[cite: 20]
  │   • Acoustic: 8,000-sample chunks (500 ms) -> wav2vec2-base -> Linear -> (T, 128)[cite: 20]
  │   • Linguistic: SentenceTransformer -> Utterance-level Sentence Embeddings -> Zero-Order Hold -> (T, 128)[cite: 20]
  └── Slicing: Harmonized Windows: T = 10 (5.0s), K = 10 (5.0s), H = 10 (5.0s, eval @ 4, 6, 10), Stride = 1[cite: 29]
      Output Slices: data/processed/omg/vid_XXXX_clip_YYYYY.npz
                     │
                     ▼
  [STAGE 2C: Manifest Compilation & Official Split Preservation (datasets/manifests/build_omg_manifest.py)]
  ├── Dialogue/Speaker Independence: Official Challenge Partition (Train: 118, Val: 60, Test: 60 videos)[cite: 27]
  └── Emits: omg_train_manifest.csv, omg_val_manifest.csv, omg_test_manifest.csv
                     │
                     ▼
  [STAGE 2D: Universal Disk Dataset Loader (datasets/loaders/omg_disk_dataset.py)]
  └── Emits Invariant Universal Batch Contract:
      {
        "modalities": { "video": (B, T, 128), "audio": (B, T, 128), "text": (B, T, 128) },
        "history_continuous": (B, K, 2),
        "history_discrete":   (B, K, 1),   # Optional 7-class discrete emotion[cite: 25]
        "recognition_targets": (B, T, 2),
        "forecasting_targets": (B, H, 2)
      }[cite: 20, 29]
                     │
                     ▼
  [STAGE 2E: Integration Pipeline Smoke Test (tests/test_omg_pipeline.py)]
  └── Validates: Stacked (B, T, 3, 128) -> Fusion (256) -> History -> Transformer -> Dual Heads[cite: 20, 29]
                     │
                     ▼
  [STAGE 2F: Model Benchmarking Matrix (training/train_omg.py)]
  ├── Tier 1 (Mandatory Core Matrix): Baseline B2 -> Baseline B1 -> Baseline B3 (4Q) -> QSAL (4Q)[cite: 30, 31]
  ├── Tier 2 (Target Scaling Matrix): B3 (6Q, 8Q) and QSAL (6Q, 8Q)[cite: 20, 30]
  └── Multi-Task Objectives: CCC(rec) + RMSE(fore) over horizons H in {4, 6, 10}[cite: 27, 29]
                     │
                     ▼
  [STAGE 2G: History Capacity Sensitivity Ablation (Optional / Time-Budgeted)]
  └── Evaluates expressive momentum: History OFF (H=0) vs. Ratio in {8, 4, 2} on behavioral tokens[cite: 29]
========================================================================================================================
```

---

## 2. Invariant Architectural Principles

To ensure zero architectural churn across the thesis, Phase 2 strictly inherits the modular contracts established in Phase 1:

1. **Model Invariance:** The core execution engine is strictly `EmotionForecastingFramework`. The repository contains **no** `OmgModel` or `CaseModel` classes.
2. **Modality Ingestion Invariance:** The modality stacker receives \(M = 3\) active channels (`video`, `audio`, `text`) instead of \(M = 8\) physiological streams. Each encoder emits an identical \((B, T, 128)\) tensor.
3. **Decoupled Quantum Register:** Variational quantum circuits (B3 and QSAL) operate on a fixed register width \(Q \in \{4, 6, 8\}\), completely decoupled from the active modality count \(M = 3\).
4. **Standardized Temporal Horizon:** Synchronized to the universal **\(2\text{ Hz}\) (\(\Delta t = 500\text{ ms}\))** operational standard (\(T = 10\), \(K = 10\), \(H = 10\), \(\text{stride} = 1\)).

---

## 3. Methodological Alternatives, Trade-offs, & Recommendations

### 3.1 Continuous Trajectory Reconstruction from Sparse Utterances

```text
Raw Dialogue:   |--- Utterance 1 (4.0s) ---|   Pause (1.0s)   |------ Utterance 2 (5.0s) ------|[cite: 22, 24]
Utterance Mid:             t_1                                               t_2
EWE Labels:             (v_1, a_1)                                        (v_2, a_2)[cite: 25, 26]
                            │                                                 │
                            ▼                                                 ▼
PCHIP Spline:   ────────────●─────────────────────────────────────────────────●───────────[cite: 29, 30]
                            │                                                 │
Sampled @ 2 Hz: ───┬────┬───┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┼───┬─────── (dt = 0.5s)
Time Steps (t):   t=1  t=2 t=3  t=4  t=5  t=6  t=7  t=8  t=9 t=10 t=11 t=12 t=13 t=14
```

- **Alternative 1: Constant Step-Function / Zero-Order Hold**
  - **Method:** Replicate the utterance-level scalar across every \(500\text{ ms}\) frame within that utterance's duration.
  - **Pros:** No synthetic interpolated points are introduced.
  - **Cons:** Creates mathematical step discontinuities \((dy/dt \to \infty)\) at segment boundaries. Destroys derivative tracking in the causal GRU and collapses multi-step forecasting into constant step replication.

- **Alternative 2: Natural Cubic Splines**
  - **Method:** Fit continuous cubic polynomials across utterance midpoints.
  - **Pros:** Generates \(\mathcal{C}^2\)-continuous smooth trajectories.
  - **Cons:** Prone to Runge's phenomenon and overshoot, oscillating outside the valid \([-1, 1]\) target range.

- **Alternative 3: Monotonic PCHIP Interpolation (RECOMMENDED)**
  - **Method:** Fit Piecewise Cubic Hermite Interpolating Polynomials (`scipy.interpolate.PchipInterpolator`) through utterance midpoint anchors \(t_i = \frac{t_{\text{start}, i} + t_{\text{end}, i}}{2}\) and resample at \(2\text{ Hz}\).
  - **Pros:** Preserves local monotonicity, ensures zero artificial overshoot, and provides continuous first derivatives (\(\mathcal{C}^1\)) necessary for multi-step forecasting.
  - **Cons:** Requires generating reconstructed targets, which must be clearly documented in the thesis as interpolated targets.

---

### 3.2 Visual Modality Encoding (\(25\text{--}30\text{ fps} \to 2\text{ Hz}\))

- **Alternative 1: Center-Frame Subsampling**
  - **Method:** Select the single video frame closest to the center of each \(500\text{ ms}\) bin and pass it through a 2D CNN.
  - **Pros:** Lightweight and rapid preprocessing.
  - **Cons:** Highly vulnerable to momentary facial occlusions, eye blinks, and head-turn artifacts.

- **Alternative 2: 3D Spatio-Temporal Backbones (VideoMAE / TimeSformer)**
  - **Method:** Ingest raw video volumes into a 3D video transformer.
  - **Pros:** Models fine-grained temporal micro-dynamics directly from pixel tubes.
  - **Cons:** Huge computational and memory footprint; parameter count breaks the capacity-matching baseline constraints.

- **Alternative 3: 2D CNN with 500 ms Temporal Feature Pooling (RECOMMENDED)**
  - **Method:** Detect face bounding boxes across all frames in each \(500\text{ ms}\) bin (\(\sim 12\text{--}15\text{ frames}\)), extract feature vectors using frozen `torchvision.models.resnet18` pretrained on **AffectNet**, average the vectors across the bin, and project linearly to \(128\text{-D}\):

  \[
  \mathbf{e}_{\text{video}}(t) =
  \operatorname{LayerNorm}\left(\mathbf{W}_{\text{video}} \cdot \bar{\mathbf{f}}_{\text{pool}}(t)\right)
  \in \mathbb{R}^{128}
  \]

  - **Pros:** Suppresses frame jitter and head motion blur while strictly matching the \((B, T, 128)\) token contract.
  - **Cons:** Requires running face extraction across all video frames during offline preprocessing.

---

### 3.3 Acoustic Modality Encoding (\(16\text{ kHz} \to 2\text{ Hz}\))

- **Alternative 1: Handcrafted Acoustic Low-Level Descriptors (openSMILE / eGeMAPS)**
  - **Method:** Extract 88 statistical functionals (MFCCs, jitter, shimmer, HNR) over \(500\text{ ms}\) frames.
  - **Pros:** Compact feature vectors and fast extraction.
  - **Cons:** Handcrafted heuristics lack deep acoustic-phonetic representations, limiting cross-modal attention capacity.

- **Alternative 2: Self-Supervised Waveform Encoding via wav2vec2-base (RECOMMENDED)**
  - **Method:** Slice audio into synchronized \(500\text{ ms}\) chunks (\(8{,}000\text{ samples}\) at \(16\text{ kHz}\)), feed into frozen `facebook/wav2vec2-base-960h`, pool temporally across convolutional representations, and project \(768 \to 128\):

  \[
  \mathbf{e}_{\text{audio}}(t) =
  \operatorname{LayerNorm}\left(\mathbf{W}_{\text{audio}} \cdot \bar{\mathbf{f}}_{\text{w2v}}(t)\right)
  \in \mathbb{R}^{128}
  \]

  - **Pros:** SOTA speech emotion representations, fully open-source (HuggingFace), directly specified in the approved thesis exposé.
  - **Cons:** Higher GPU memory footprint during offline extraction compared to openSMILE.

---

### 3.4 Linguistic Modality Encoding (ASR Transcripts \(\to 2\text{ Hz}\))

- **Alternative 1: Forced-Alignment Word Slicing (MFA / Whisper-CTC)**
  - **Method:** Align word embeddings strictly to the fractions of a second during which individual words are vocalized.
  - **Pros:** Exact sub-second lexical synchronization.
  - **Cons:** Produces extreme sparsity where many \(500\text{ ms}\) bins contain null tokens, destabilizing cross-modal attention gradients.

- **Alternative 2: Utterance-Level Zero-Order Hold with Learned Silence Token (RECOMMENDED)**
  - **Method:** Embed each utterance transcript using `sentence-transformers/all-MiniLM-L6-v2`, projected to \(128\text{-D}\). Propagate this context vector across all \(500\text{ ms}\) steps overlapping \([t_{\text{start}, i}, t_{\text{end}, i}]\), assigning a learned silence token \(\mathbf{e}_{\text{silence}} \in \mathbb{R}^{128}\) during non-speech pauses.
  - **Pros:** Standard practice in multimodal affective benchmarks (CMU-MOSEI, MuSe, AVEC); reflects the cognitive reality that semantic conversational context persists in working memory throughout a spoken thought.
  - **Cons:** Coarser temporal resolution than word-aligned models.

---

### 3.5 Feature Extraction & Storage Strategy

- **Alternative 1: On-the-Fly Feature Extraction During Training**
  - **Method:** Decode video, crop faces, and execute wav2vec2 inside PyTorch's `__getitem__()`.
  - **Pros:** Eliminates intermediate processed storage.
  - **Cons:** Major training bottleneck; leads to GPU starvation and multi-week training runs.

- **Alternative 2: Offline Extraction & Static `.npz` Slicing (RECOMMENDED)**
  - **Method:** Run ResNet18, wav2vec2, and MiniLM extraction once offline, saving pre-aligned \((T, 128)\) tensors directly inside `data/processed/omg/vid_XXXX_clip_YYYYY.npz`.
  - **Pros:** Fast data loading matching CASE; ensures identical multi-epoch training throughput.
  - **Cons:** Requires upfront disk storage for cached feature arrays.

---

## 4. Stage-by-Stage Implementation Blueprint

### Stage 2A: Pre-flight Audit & Dialogue Inspection

- **Script:** `tools/inspect_omg_dataset.py`
- **Checks:**
  1. Parse all 567 raw dialogue JSON/CSV files and verify valid video containers.
  2. Audit timestamp consistency: verify \(t_{\text{start}, i} < t_{\text{end}, i} \le t_{\text{start}, i+1}\) across all 7,371 utterances.
  3. Validate audio track continuity and verify non-empty transcript strings.
  4. Measure ground-truth EWE continuous distributions (Valence in \([-1, 1]\), Arousal in \([0, 1]\)) and 7-class discrete distributions.
- **Outputs:** `reports/omg_statistics.json` and `docs/datasets/omg_dataset_spec.md`.

---

### Stage 2B: Offline Preprocessing & Feature Caching

- **Script:** `datasets/preprocessing/omg_preprocessor.py`
- **Execution:**
  1. Fit monotonic PCHIP splines across utterance midpoint anchors and resample at \(2\text{ Hz}\) (\(\Delta t = 500\text{ ms}\)).
  2. Extract video features via `torchvision.models.resnet18` pretrained on **AffectNet**, with \(500\text{ ms}\) temporal pooling to \((T, 128)\).
  3. Extract audio features via `facebook/wav2vec2-base-960h` over \(8{,}000\)-sample slices to \((T, 128)\).
  4. Extract text features via `sentence-transformers/all-MiniLM-L6-v2` with zero-order hold and learned silence masking to \((T, 128)\).
  5. Slice into sliding windows (\(T = 10\), \(K = 10\), \(H = 10\), \(\text{stride} = 1\)) and save compressed `.npz` files.

---

### Stage 2C: Manifest Compilation & Split Preservation

- **Script:** `datasets/manifests/build_omg_manifest.py`
- **Partition Protocol:** Reads official Challenge video IDs, partitioning slices strictly by video:
  - **Train Set:** 118 monologues.
  - **Validation Set:** 60 monologues.
  - **Test Set:** 60 monologues.
- **Outputs:** Emits `omg_train_manifest.csv`, `omg_val_manifest.csv`, and `omg_test_manifest.csv`.

---

### Stage 2D: Universal Disk Dataset Loader

- **Script:** `datasets/loaders/omg_disk_dataset.py`
- **Mechanism:** Reads precomputed `.npz` slices indexed by manifest CSVs without runtime video decoding.
- **Output Contract:**

```python
batch = {
    "modalities": {
        "video": torch.Tensor,       # (B, T=10, 128)
        "audio": torch.Tensor,       # (B, T=10, 128)
        "text":  torch.Tensor        # (B, T=10, 128)
    },
    "history_continuous": torch.Tensor,  # (B, K=10, 2) [Valence, Arousal]
    "history_discrete":   torch.Tensor,  # (B, K=10, 1) [Optional 7-class emotion]
    "recognition_targets": {
        "continuous": torch.Tensor       # (B, T=10, 2)
    },
    "forecasting_targets": {
        "continuous": torch.Tensor       # (B, H=10, 2)
    },
    "metadata": {
        "video_id": str,
        "clip_idx": int
    }
}
```

---

### Stage 2E: Integration Pipeline Smoke Test

- **Script:** `tests/test_omg_pipeline.py`
- **Verification Scope:**
  1. Stacks the 3 modality tokens into \((B, T, 3, 128)\).
  2. Verifies that the fusion layer dynamically adapts from \(M = 8\) (CASE) to \(M = 3\) (OMG), projecting to \((B, T, 256)\) without structural code changes.
  3. Verifies post-fusion history conditioning (\(\text{Ratio} = 4 \implies H = 64\text{-D}\)) and sequence length preservation (\((B, T, 256)\) entering Transformer).
  4. Confirms non-autoregressive decoding over \(H = 10\) steps and validates gradient backpropagation across classical and quantum layers.

---

### Stage 2F: Benchmarking Matrix & Prioritized Execution Order

- **Script:** `training/train_omg.py`
- **Configuration:** `configs/experiments/omg_benchmark.yaml`

```text
                                  OMG EXECUTION PRIORITY TIERS
                                                │
         ┌──────────────────────────────────────┴──────────────────────────────────────┐
         ▼                                                                             ▼
  [TIER 1: MANDATORY CORE MATRIX]                                       [TIER 2: TARGET SCALING & ABLATIONS]
  1. Baseline B2 (Unified Concat)                                       1. Quantum Capacity Scaling:
  2. Baseline B1 (Classical Self-Attention)                                • Baseline B3: 6Q, 8Q
  3. Baseline B3 (Entangled PQC Pooling, Q = 4)                            • Proposed QSAL: 6Q, 8Q
  4. Proposed QSAL (Quantum Self-Attention, Q = 4)                      2. History Ratio Sensitivity Ablation:
     (All evaluated under default Ratio = 4)                               • History OFF (H = 0)
                                                                           • Minimalist (Ratio = 8 -> H = 32-D)
                                                                           • High Capacity (Ratio = 2 -> H = 128-D)
```

#### Multi-Task Optimization Loss

\[
\mathcal{L}_{\text{total}} =
\lambda_{\text{rec}} \left(1 - \operatorname{CCC}(\mathbf{Y}_{\text{rec}}, \hat{\mathbf{Y}}_{\text{rec}})\right)
+ \lambda_{\text{fore}} \operatorname{RMSE}(\mathbf{Y}_{\text{fore}}, \hat{\mathbf{Y}}_{\text{fore}})
\]

**Reported metrics:** Frame-level recognition (\(\text{CCC}_{\text{val}}\), \(\text{CCC}_{\text{aro}}\)) and multi-step trajectory forecasting evaluated at short (\(H = 4 \implies 2.0\text{ s}\)), medium (\(H = 6 \implies 3.0\text{ s}\)), and long (\(H = 10 \implies 5.0\text{ s}\)) horizons.

---

## 5. Deliverable: Thesis Chapter 5.2 Results Table

Executing this matrix populates the empirical benchmark table for **Section 5.2 (Expressive Behavioral Domain Results)**.

| Model Index | Model Name | Fusion Mechanism | Qubit Width (\(Q\)) | History Ratio (\(H\)) | Execution Tier | Recognition \(\text{CCC}_{\text{val}} \uparrow\) | Recognition \(\text{CCC}_{\text{aro}} \uparrow\) | Forecast \(\text{CCC} (H=4)\) | Forecast \(\text{CCC} (H=6)\) | Forecast \(\text{CCC} (H=10)\) | Overall \(\text{RMSE} \downarrow\) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **M1** | Baseline B2 | Direct Concat | — | \(4:1\) (\(64\)) | Tier 1 (Core) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M2** | Baseline B1 | Classical Attn | — | \(4:1\) (\(64\)) | Tier 1 (Core) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M3** | Baseline B3 | PQC Pooling | \(4\) | \(4:1\) (\(64\)) | Tier 1 (Core) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M4** | Proposed QSAL | Quantum Attn | \(4\) | \(4:1\) (\(64\)) | Tier 1 (Core) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M5** | Baseline B3 | PQC Pooling | \(6\) | \(4:1\) (\(64\)) | Tier 2 (Scaling) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M6** | Baseline B3 | PQC Pooling | \(8\) | \(4:1\) (\(64\)) | Tier 2 (Scaling) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M7** | Proposed QSAL | Quantum Attn | \(6\) | \(4:1\) (\(64\)) | Tier 2 (Scaling) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M8** | Proposed QSAL | Quantum Attn | \(8\) | \(4:1\) (\(64\)) | Tier 2 (Scaling) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M9** | QSAL Ablation | Quantum Attn | \(4\) | Disabled (\(0\)) | Tier 2 (Ablation) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M10** | QSAL Ablation | Quantum Attn | \(4\) | \(8:1\) (\(32\)) | Tier 2 (Ablation) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |
| **M11** | QSAL Ablation | Quantum Attn | \(4\) | \(2:1\) (\(128\)) | Tier 2 (Ablation) | *eval* | *eval* | *eval* | *eval* | *eval* | *eval* |

---

## 6. Framework Invariance & Decision Freeze Summary

| Parameter / Component | Status | Justification & Scope |
| --- | --- | --- |
| 1. Temporal Clock (\(dt\)) | **FROZEN** | Standardized at 2.0 Hz (500 ms) across CASE & OMG[cite: 30]. |
| 2. Window Bounds (\(T, K, H\)) | **FROZEN** | \(T=10\) (5s), \(K=10\) (5s), \(H=10\) (eval @ 4, 6, 10), stride=1[cite: 29]. |
| 3. Target Continuous Space | **FROZEN** | Standardized to [valence, arousal][cite: 29, 30]. |
| 4. Label Trajectory Interpolator | **FROZEN** | Monotonic PCHIP across utterance midpoints[cite: 26, 29]. |
| 5. Text Alignment Paradigm | **FROZEN** | Utterance persistence (Zero-Order Hold) + silence[cite: 29]. |
| 6. Batch Data Contract | **FROZEN** | Standard dictionary emitting \((B, T, 128)\) per modality[cite: 20]. |
| 7. Preprocessing Storage | **FROZEN** | Offline pre-extracted `.npz` feature caching[cite: 20, 30]. |
| 8. Pretrained Feature Backbones | **FROZEN (Concrete)** | Video: ResNet18 (AffectNet); Audio: wav2vec2-base; Text: SentenceTransformer (`all-MiniLM-L6-v2`). |
| 9. Benchmark Scaling (6Q, 8Q) | **TIER 2 (Time-Budget)** | Mandatory core: 4Q (B2, B1, B3, QSAL). Scale if time permits[cite: 30]. |

---

## 7. Finalized Feature Backbone Configuration

The following feature backbone configuration is frozen for Phase 2 and replaces any earlier generic backbone descriptions.

```yaml
video:
  backbone: ResNet18
  pretraining: AffectNet
  output_dim: 128

audio:
  backbone: wav2vec2-base
  output_dim: 128

text:
  backbone: SentenceTransformer
  output_dim: 128
```
