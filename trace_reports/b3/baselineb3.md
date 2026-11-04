# BaselineB3 Architecture Audit

Generated: 2026-10-25 21:12:11.376157

## Configuration

- **History Length**: 50
- **Forecast Horizon**: 25
- **Qubits**: 4
- **Circuit Depth**: 3
- **Transformer Dimension**: 256
- **Backend**: lightning.qubit

## Runtime Module Execution

1. MultimodalFeatureExtractorsV2
2. UnifiedPQC
3. BaselineB3PQCFusion
4. TransformerEncoderDecoder
5. RecognitionHead
6. ForecastingHead

## Detailed Audit

## Raw Modalities

### audio

**Input:** `-`

**Output:** `(1, 50, 2)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Raw Modalities

### bpm

**Input:** `-`

**Output:** `(1, 50, 1)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Raw Modalities

### ecg

**Input:** `-`

**Output:** `(1, 50, 1)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Raw Modalities

### eda

**Input:** `-`

**Output:** `(1, 50, 1)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Raw Modalities

### eeg

**Input:** `-`

**Output:** `(1, 50, 1)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Raw Modalities

### emg

**Input:** `-`

**Output:** `(1, 50, 1)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Raw Modalities

### eye

**Input:** `-`

**Output:** `(1, 50, 2)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Raw Modalities

### resp

**Input:** `-`

**Output:** `(1, 50, 1)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Raw Modalities

### text

**Input:** `-`

**Output:** `(1, 50, 2)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Raw Modalities

### video

**Input:** `-`

**Output:** `(1, 50, 2)`

**Purpose:** Synchronized input modality

**Status:** PASS

## Runtime

### MultimodalFeatureExtractorsV2

**Input:** `[{'audio': (1, 50, 2), 'bpm': (1, 50, 1), 'ecg': (1, 50, 1), 'eda': (1, 50, 1), 'eeg': (1, 50, 1), 'emg': (1, 50, 1), 'eye': (1, 50, 2), 'resp': (1, 50, 1), 'text': (1, 50, 2), 'video': (1, 50, 2)}]`

**Output:** `{'ecg': (1, 50, 128), 'eda': (1, 50, 128), 'eeg': (1, 50, 128), 'emg': (1, 50, 128), 'resp': (1, 50, 128), 'bpm': (1, 50, 128), 'eye': (1, 50, 128), 'audio': (1, 50, 128), 'video': (1, 50, 128), 'text': (1, 50, 128)}`

**Purpose:** Runtime module execution

**Status:** PASS

## Runtime

### UnifiedPQC

**Input:** `[(1, 50, 4)]`

**Output:** `(1, 50, 4)`

**Purpose:** Runtime module execution

**Status:** PASS

## Runtime

### BaselineB3PQCFusion

**Input:** `[{'ecg': (1, 50, 128), 'eda': (1, 50, 128), 'eeg': (1, 50, 128), 'emg': (1, 50, 128), 'resp': (1, 50, 128), 'bpm': (1, 50, 128), 'eye': (1, 50, 128), 'audio': (1, 50, 128), 'video': (1, 50, 128), 'text': (1, 50, 128)}]`

**Output:** `(1, 50, 256)`

**Purpose:** Runtime module execution

**Status:** PASS

## Runtime

### TransformerEncoderDecoder

**Input:** `[]`

**Output:** `[(1, 50, 256), (1, 25, 256)]`

**Purpose:** Runtime module execution

**Status:** PASS

## Runtime

### RecognitionHead

**Input:** `[(1, 50, 256)]`

**Output:** `{'continuous': {'valence': (1,), 'arousal': (1,), 'dominance': (1,), 'stress': (1,), 'workload': (1,), 'fatigue': (1,)}, 'discrete': {'emotion_class': (1, 6), 'stress_class': (1, 3), 'workload_class': (1, 3), 'alertness_class': (1, 3), 'drowsiness_class': (1, 4), 'kss_score': (1, 9)}}`

**Purpose:** Runtime module execution

**Status:** PASS

## Runtime

### ForecastingHead

**Input:** `[(1, 25, 256)]`

**Output:** `{'continuous': {'valence': (1, 25), 'arousal': (1, 25), 'dominance': (1, 25), 'stress': (1, 25), 'workload': (1, 25), 'fatigue': (1, 25)}, 'discrete': {'emotion_class': (1, 25, 6), 'stress_class': (1, 25, 3), 'workload_class': (1, 25, 3), 'alertness_class': (1, 25, 3), 'drowsiness_class': (1, 25, 4), 'kss_score': (1, 25, 9)}}`

**Purpose:** Runtime module execution

**Status:** PASS

## Final Outputs

### Fusion Output

**Input:** `-`

**Output:** `(1, 50, 256)`

**Theory:** (B,T,256)

**Purpose:** Transformer-ready fused representation

**Status:** PASS

## Final Outputs

### Transformer Encoder

**Input:** `-`

**Output:** `(1, 50, 256)`

**Theory:** (B,T,256)

**Purpose:** Temporal encoded sequence

**Status:** PASS

## Final Outputs

### Transformer Decoder

**Input:** `-`

**Output:** `(1, 25, 256)`

**Theory:** (B,H,256)

**Purpose:** Future latent sequence

**Status:** PASS


## Summary

- PASS: 19
- FAIL: 0
- RESULT: PASS