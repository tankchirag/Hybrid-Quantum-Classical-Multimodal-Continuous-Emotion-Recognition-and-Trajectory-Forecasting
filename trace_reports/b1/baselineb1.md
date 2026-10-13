# BaselineB1 Architecture Audit

Generated: 2026-10-08 21:54:09.138523

## Configuration

- **History Length**: 50
- **Forecast Horizon**: 25
- **Fusion Dimension**: 64
- **Transformer Dimension**: 256
- **Attention Heads**: 4

## Runtime Module Execution

1. MultimodalFeatureExtractorsV2
2. Tokenizer
3. ClassicalFusion
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

### Tokenizer

**Input:** `[{'ecg': (1, 50, 128), 'eda': (1, 50, 128), 'eeg': (1, 50, 128), 'emg': (1, 50, 128), 'resp': (1, 50, 128), 'bpm': (1, 50, 128), 'eye': (1, 50, 128), 'audio': (1, 50, 128), 'video': (1, 50, 128), 'text': (1, 50, 128)}]`

**Output:** `(1, 50, 10, 128)`

**Purpose:** Runtime module execution

**Status:** PASS

## Runtime

### ClassicalFusion

**Input:** `[(1, 50, 10, 128)]`

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

**Output:** `{'continuous': {'valence': (1,), 'arousal': (1,), 'dominance': (1,), 'stress': (1,), 'workload': (1,), 'fatigue': (1,)}, 'discrete': {}}`

**Purpose:** Runtime module execution

**Status:** PASS

## Runtime

### ForecastingHead

**Input:** `[(1, 25, 256)]`

**Output:** `{'continuous': {'valence': (1, 25), 'arousal': (1, 25), 'dominance': (1, 25), 'stress': (1, 25), 'workload': (1, 25), 'fatigue': (1, 25)}, 'discrete': {}}`

**Purpose:** Runtime module execution

**Status:** PASS

## Final Outputs

### Token Output

**Input:** `-`

**Output:** `(1, 50, 10, 128)`

**Theory:** (B,T,N,128)

**Purpose:** Tokenized modality representations

**Status:** PASS

## Final Outputs

### Fusion Output

**Input:** `-`

**Output:** `(1, 50, 256)`

**Theory:** (B,T,256)

**Purpose:** Classical self-attention fused representation

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

- PASS: 20
- FAIL: 0
- RESULT: PASS