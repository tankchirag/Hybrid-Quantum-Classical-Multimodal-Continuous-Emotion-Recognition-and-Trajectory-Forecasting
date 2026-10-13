# Theory vs Implementation

## extractors

**Theory:** Required by theory

**Runtime:** Found

**Status:** PASS

## fusion

**Theory:** Required by theory

**Runtime:** Found

**Status:** PASS

## q_pqc

**Theory:** Required by theory

**Runtime:** Found

**Status:** PASS

## k_pqc

**Theory:** Required by theory

**Runtime:** Found

**Status:** PASS

## v_pqc

**Theory:** Required by theory

**Runtime:** Found

**Status:** PASS

## attention

**Theory:** Required by theory

**Runtime:** Found

**Status:** PASS

## temporal_model

**Theory:** Required by theory

**Runtime:** Found

**Status:** PASS

## recognition_head

**Theory:** Required by theory

**Runtime:** Found

**Status:** PASS

## forecasting_head

**Theory:** Required by theory

**Runtime:** Found

**Status:** PASS

## MultimodalFeatureExtractorsV2

**Theory:** (B,T,128)

**Runtime:** {'ecg': (1, 50, 128), 'eda': (1, 50, 128), 'eeg': (1, 50, 128), 'emg': (1, 50, 128), 'resp': (1, 50, 128), 'bpm': (1, 50, 128), 'eye': (1, 50, 128), 'audio': (1, 50, 128), 'video': (1, 50, 128), 'text': (1, 50, 128)}

**Status:** PASS

## Q-PQC

**Theory:** (B*T*N,1,Q)

**Runtime:** (500, 1, 6)

**Status:** PASS

## K-PQC

**Theory:** (B*T*N,1,Q)

**Runtime:** (500, 1, 6)

**Status:** PASS

## V-PQC

**Theory:** (B*T*N,1,Q)

**Runtime:** (500, 1, 6)

**Status:** PASS

## GaussianAttention

**Theory:** (B*T,N,Q)

**Runtime:** (50, 10, 6)

**Status:** PASS

## QSALFusion

**Theory:** (B,T,256)

**Runtime:** (1, 50, 256)

**Status:** PASS

## TransformerEncoderDecoder

**Theory:** (B,T,256)->(B,H,256)

**Runtime:** [(1, 50, 256), (1, 25, 256)]

**Status:** PASS

## RecognitionHead

**Theory:** Recognition Outputs

**Runtime:** {'continuous': {'valence': (1,), 'arousal': (1,), 'dominance': (1,), 'stress': (1,), 'workload': (1,), 'fatigue': (1,)}, 'discrete': {}}

**Status:** PASS

## ForecastingHead

**Theory:** Forecast Outputs

**Runtime:** {'continuous': {'valence': (1, 25), 'arousal': (1, 25), 'dominance': (1, 25), 'stress': (1, 25), 'workload': (1, 25), 'fatigue': (1, 25)}, 'discrete': {}}

**Status:** PASS
