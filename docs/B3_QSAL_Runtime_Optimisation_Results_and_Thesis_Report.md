# B3 and QSAL Runtime Optimisation: Methods, Verification, Results and Reproducibility

**Project:** Hybrid Quantum–Classical Multimodal Continuous Emotion Recognition and Trajectory Forecasting  
**Study:** CASE Phase 1 — B3 unified-PQC fusion and QSAL quantum self-attention  
**Document type:** Thesis-ready experimental report and reproducibility README  
**Evidence cutoff:** 22 November 2026  
**Implementation branch:** `perf/b3-qsal-runtime-opt`  
**Completed optimised benchmark commit:** `0425b741b5b78abbf513f6f586dd986ae7bc7aab`  
**Principal complete-epoch experiment:** SLURM job **8306** (21 November 2026)  
**Status:** Optimised one-epoch experiments complete; matched full-epoch reference and long-run learning equivalence not established  
**Revision:** v1.1 — explicit job-ID ledger, numerical reconciliation, comparison tables and thesis-citation checklist

> **Document control.** This is a *results and interpretation companion* to `docs/B3_QSAL_Runtime_Optimisation_Technical_Record.md`. **Do not replace or delete** that earlier record: it preserves the original plan, decisions, code audits, failed attempts and historical evidence. The older record contains some now-obsolete “next actions” (e.g., job 7742); those passages remain historical, not current operating instructions. Use this document for the consolidated post-implementation narrative. Original experiment artifacts and logs are authoritative if numbers disagree.

---

## Navigation / where to find the evidence

| Research question | Main sections |
|---|---|
| What was prohibitively slow before optimisation? | §§2, 9, 18 |
| What optimisation strategies were planned, adopted or deferred? | §§3–5, 11 |
| Which specific jobs verified batching and numerical correctness? | §§4–5, 17 |
| What were the original and optimised times? | §§2, 4–7, 9, 18 |
| Can the results be described as a measured end-to-end speedup? | §§9, 14, 18 |
| Where are the script, YAML, evidence and checkpoints? | §§7–8, 12–13, 17, 19 |
| What can be included in the thesis? | §§14–15, 20 |

---

## Executive summary

The original CASE B3 and QSAL implementations were impractical to train extensively under the early runtime observations. The historical Phase 1 report listed **15.148 s/batch (B3)** and **32.080 s/batch (QSAL)** and reported projections of **15.02 h/epoch** and **31.81 h/epoch**. However, the historical batch-size and epoch-hour arithmetic are inconsistent; these projections **must not be treated as verified measured epoch runtimes** or directly compared to job 8306 to claim a controlled speedup. Moreover, early QSAL reference job 7898 had a confirmed one-modality source-propagation defect, so it is not a valid eight-modality QSAL reference.

The runtime optimisation programme focused on implementation cost, not changing the scientific models. Two changes were adopted:

1. **Broadcast / batch the existing parameterised quantum circuit (PQC) inputs in `quantum/unified_pqc.py`.** Instead of invoking an existing QNode once for every flattened sample/time input, pass the flattened angle batch to the same QNode. The bounded T=2 profile reduced Python-level QNode invocations from **16 to 1 for B3** and from **384 to 3 for QSAL** (one per independent Q, K and V branch). With the *same* `lightning.qubit`/`best` settings and bounded synthetic profile, forward-pass medians improved **1.317× (B3)** and **1.452× (QSAL)**. This isolates the batching change better than the later comparisons.
2. **Use PennyLane `default.qubit` with `diff_method=backprop` (Torch interface, analytic shots)** instead of the reference `lightning.qubit` with `diff_method=best`. Paired real-CASE, one-optimizer-step comparisons measured **8.126 vs 1.375 s for B3** (**5.91×**) and **184.919 vs 0.511 s for QSAL** (approximately **362×**), with verified output/gradient/update agreement. These are **single-step wall-time ratios, not steady-state or full-epoch acceleration factors**.

The adopted candidate subsequently ran the *actual* CASE train and validation workloads in **SLURM job 8306**, on **one full NVIDIA A100 80GB PCIe GPU**, using **batch size 8**, **28,560 training windows**, **7,140 validation windows**, and **T=120/H=20**. Including initialization, training, validation, checkpoint creation and process finalization, the measured one-epoch entrypoint durations were **451 s (7m31s) for B3** and **2,164 s (36m04s) for QSAL**. Both experiments completed, wrote `best.pt`/`last.pt`, and produced finite train/validation losses. The final operational duration ratio was **QSAL/B3 = 4.798×**.

**Scientific conclusion:** Batching reduces QNode dispatch; the differentiable simulator/backend choice makes full-length quantum training operationally feasible; the architecture was preserved in the validated tests. The *causal contribution of each change* is only established within its specified paired test. A **matched reference-versus-candidate full-epoch speedup** and **multi-epoch/multi-seed learning equivalence** have not been demonstrated.

---

## 1. Research motivation and scientific constraints

### 1.1 Why this optimisation was necessary

Hybrid quantum–classical models combine conventional neural-network training with differentiable circuit simulation. In these architectures, many four-qubit PQCs are evaluated for time-dependent or modality-dependent representations. Although each individual four-qubit statevector is small, repeated framework dispatch, backend overhead and quantum differentiation can dominate end-to-end computation. This prevented practical, multi-epoch experimentation despite modest model parameter counts (~8 million).

### 1.2 Architectures and invariant comparisons

| Property | B3 | QSAL |
|---|---|---|
| Role | Unified entangled-PQC multimodal fusion | Quantum Q/K/V generation with Gaussian self-attention over modality tokens |
| Quantum modules | One unified PQC | Three independently parameterised PQCs (Q, K and V) |
| Qubits / circuit depth | 4 / 3 | 4 / 3 in each branch |
| Modality inputs | Eight configured CASE physiology modalities | The same eight, preserved in configured order |
| Quantum output | Fused representation for downstream temporal model | Q/K/V-derived multimodal attention features for downstream temporal model |
| Shared representation contract | `(B,T,256)` fusion output | `(B,T,256)` fusion output |
| Continuous targets | Valence and arousal recognition and forecasting | Same |

**Common controls:** physiological encoders, history representation and conditioning, temporal backbone, recognition and forecasting heads, loss, optimisation settings, train/validation/test split, data normalization, label handling, batch size, seed and precision should remain consistent when isolating an implementation speed change.

**Completed CASE setup:** ECG, BVP, GSR, RSP, SKT, EMG-ZYGO, EMG-CORU, EMG-TRAP; 2 Hz; observation/history **120 steps** (60 s); forecasting **20 steps** (10 s); 30 subjects split train 1–20, validation 21–25, test 26–30; subject-independent split; stride 1. The configured quantum simulation has `shots: null` (analytic), not hardware shots.

**Not permissible as silent optimisations:** dropping modalities; reusing Q, K or V parameters; changing circuit depth, qubit width, entanglement, gate sequence, encoding, measurements, attention equation, loss or forecasting horizon; shortening input sequences; changing the effective dataset. Such changes require separately labelled ablations and renewed statistical evaluation.

### 1.3 Corrected QSAL input selection — distinct from runtime optimisation

The technical record documents a **pre-optimisation QSAL defect**: QSAL constructed `QSALFusion` without the configured sources, so a legacy default selected ECG only, despite eight configured modalities. The wrapper was corrected to propagate the ordered enabled sources, with checks for duplicate, missing and empty lists. Job **7919** verified a real batch with eight modality tensors, `(2,120,8,128)` QSAL token stack, `(2,120,256)` fusion output, and the expected recognition/forecast shapes. The parameter total remained **8,121,553**.

**Consequences for reference selection:** job **7898** was run with the defective single-ECG QSAL routing and is **not a valid full-eight-modality QSAL baseline**, regardless of its apparent speed. Preserve its original evidence but clearly exclude it from controlled performance claims. Corrected QSAL benchmarking must use all eight sources.

---

## 2. Historical problem statement and reference evidence

### 2.1 Original Phase 1 completion-report table

Source: `docs/phases/phase1_case_completion_report.md`, “Computational Benchmark”.

| Architecture | Reported seconds/batch | Reported hours/epoch | Historical interpretation |
|---|---:|---:|---|
| B2 (classical concatenation) | 0.396 | 0.39 | Low-cost classical capacity reference |
| B3 | **15.148** | **15.02** | Quantum fusion computational bottleneck |
| QSAL | **32.080** | **31.81** | Additional quantum Q/K/V cost |

The report labels the benchmark **batch size 2**, but the printed hours/epoch require scrutiny. For **28,560** train windows:

- At **batch size 2** there are **14,280** training batches; `15.148 × 14,280 / 3600 = 60.087 h` (B3) and `32.080 × 14,280 / 3600 = 127.251 h` (QSAL), **before validation and initialization**.
- At **batch size 8** there are **3,570** training batches; `15.148 × 3,570 / 3600 = 15.021 h` and `32.080 × 3,570 / 3600 = 31.813 h`. This reproduces the published 15.02/31.81 hours, **but batch size 8 contradicts the table's batch-size-2 label**.

**Decision:** Preserve the historical table exactly as a record of the initial problem, and add an explicit *unresolved denominator/protocol inconsistency*. Do not silently repair it by guessing which batch size or window count was actually used. These hour estimates are not end-to-end epoch measurements and should not be divided by job 8306's epoch durations to assert a verified speedup.

### 2.2 Additional historical reference experiments

- **B3 reference job 8026:** historical full-length batch-2 protocol, approximately **8.04 s/step** on an **RTX A6000 48 GB**, 10 warm-ups, 50 measured steps, three repetitions; approximately 24m37s of reference benchmark time. It is a step benchmark, not a complete reference epoch.
- **QSAL reference job 8236:** original `lightning.qubit` reference benchmark on an **A100 2g.20gb MIG slice** exceeded its **two-hour limit** and did **not** produce a complete comparable epoch or full timing series. It should not be rerun solely to fabricate parity with the new A100 result; the scientifically honest status is censored/incomplete.
- **Job 7898:** obsolete QSAL source-selection-defective reference; explicitly excluded from correct eight-modality comparisons.

These reference records differ in GPU type, allocated GPU fraction, microbenchmark scope or implementation correctness. They are supporting *historical context*, not a matched full-epoch comparison to job 8306.

### 2.3 Why a GPU allocation did not automatically fix the problem

The CUDA device hosting the PyTorch classical model is not necessarily the device on which the PennyLane simulator performs its circuit execution. The original path could incur host-side circuit dispatch, conversion, quantum differentiation overhead, and small repeated workloads. `nvidia-smi` alone does not prove native GPU quantum kernels. The final successful configuration establishes a valid CUDA training environment, but the result should be described as **CUDA-hosted PyTorch training with differentiable PennyLane `default.qubit`**, not as independently proven GPU-native quantum hardware acceleration.

---

## 3. Optimisation plan versus implementation status

The original record's O0–O8 scheme was a **plan**, not a list of completed interventions. Retain that distinction:

| Stage | Planned intervention | Evidence-based status as of job 8306 | Attribution / notes |
|---|---|---|---|
| O0 | Preserve reference outputs, architecture, loss and parameters | **Applied through correctness profiles/gates**, not one exhaustive frozen full-epoch O0 artifact | Historical step references remain nonuniform |
| O1 | Measure component/phase timing | **Partly measured** | T=2 forward components; full epoch train/val time; no exhaustive full-length quantum-backward breakdown |
| O2 | Count quantum QNode executions | **Measured at Python-call boundary for T=2** | Not a count of underlying simulator statevector operations |
| O3 | Remove redundant `.cpu()`, copies, device/QNode reconstruction | **Not demonstrated as performance improvement** | Original QNodes were already constructed; conversion event counts did not materially fall |
| O4 | Batch/broadcast equivalent PQC inputs | **Implemented and validated** | `quantum/unified_pqc.py`, one QNode/module per forward |
| O5 | Cache/reuse identical circuit results | **Not independently demonstrated** | Do not claim a speedup from this method |
| O6 | Screen simulator/differentiation; choose faster combination | **Implemented and validated at one step** | `lightning.qubit` + `best` → `default.qubit` + `backprop` |
| O7 | Mixed precision / AMP | **Deferred; not used** | FP32 candidate measurements |
| O8 | Multi-GPU training | **Not used** | Final benchmark used one full A100 80GB GPU |

Additional **proposals not part of the adopted candidate:** parameter chunking, manual NumPy batching, `torch.vmap`, Lightning adjoint, GPU-native Lightning simulator, shared Q/K/V parameters, 6Q/8Q variants, DataLoader tuning, and changes to source order, architecture, labels or loss.

---

## 4. Optimisation method 1 — broadcasted/batched QNode execution

### 4.1 Root cause

The old `UnifiedPQC.forward` loop flattened its batch/time inputs and called an *already-created* QNode separately for each angle vector. For QSAL, this loop repeated in independent Q-, K- and V-branches. The expense was not QNode object construction inside the loop; it was repeated **Python-to-quantum-framework invocation and associated simulator work**.

**Conceptual pseudocode (not verbatim repository code):**

```python
# Serial dispatch: logically independent inputs
results = [qnode(row) for row in angles_flat]

# Equivalent broadcasted dispatch, if device and gradients support it
results = qnode(angles_flat)
```

The same physical circuit instructions/measurements and trainable parameters are applied to the same logical inputs. An outer-batch QNode invocation count of one **does not** imply only one underlying statevector evaluation, nor does it imply no broadcasting cost.

### 4.2 Concrete isolated call-count evidence

On **T=2**, with the profiler's configured batch size 8 and eight sources:

| Model | Original QNode calls per forward | Batched calls per forward | Dispatch-call reduction |
|---|---:|---:|---:|
| B3 | **16** | **1** | **16× fewer** |
| QSAL | **384** = 128 Q + 128 K + 128 V | **3** = 1 Q + 1 K + 1 V | **128× fewer** |

**These are call-count ratios, not measured training speedups.** QSAL retains independent Q/K/V parameters and Gaussian attention.

### 4.3 Batching-only bounded performance study (jobs 7934 and 7937)

**Protocol:** original and broadcast implementations, both `lightning.qubit` + `diff_method="best"`; synthetic length T=2; four qubits, depth three, batch size 8, eight configured modalities, FP32, one A100 2g.20gb MIG allocation, one warm-up forward, three measured forwards; median reported. Not a real full epoch.

| Measure | B3 before batching | B3 after batching | QSAL before batching | QSAL after batching |
|---|---:|---:|---:|---:|
| Forward median | 89.054 ms | **67.642 ms** | 1,717.533 ms | **1,183.204 ms** |
| Forward speed ratio | — | **1.317×** | — | **1.452×** |
| Forward time saved | — | **21.412 ms** | — | **534.329 ms** |
| Forward+backward (one sample) | 1.323808 s | **0.848950 s** | 20.589187 s | **19.653934 s** |
| Forward+backward ratio (single sample) | — | **1.559×** | — | **1.048×** |

**Component medians:** B3 fusion PQC 71.772 → 51.487 ms; QSAL Q-PQC 569.863 → 386.222 ms, K-PQC 565.375 → 397.729 ms, V-PQC 565.147 → 383.797 ms. Feature extraction and temporal-backbone medians stayed broadly similar. In this bounded test, conversion events did not materially decrease: QSAL `aten::copy_` remained 16,929 calls; CUDA copy time 17.843 vs 17.945 ms. This supports **dispatch reduction**, not reduced tensor-copy count, as the observed mechanism.

### 4.4 Batching numerical correctness

Jobs 7934/7937's paired synthetic tests retained output shapes, parameter manifests, finite outputs and required gradients. Observed maximum output differences were zero; required-gradient maximum differences were approximately **3.73×10⁻⁹ (B3)** and **5.59×10⁻⁹ (QSAL)** under `atol=1e-5`, `rtol=1e-4`. B3 parameter count remained **8,073,056**; QSAL remained **8,121,553**. No optimizer step was tested by these particular T=2 batching checks.

**Caution:** the T=2 forward and backward samples cannot be multiplied by 60 to predict T=120 training behavior. The QSAL backward result in particular showed only a modest 1.048× ratio from the batching-only candidate, motivating differentiation/backend work.

---

## 5. Optimisation method 2 — differentiable simulator and backpropagation

### 5.1 Configuration change

| Aspect | Historical/batched reference | Adopted candidate |
|---|---|---|
| PennyLane device | `lightning.qubit` | **`default.qubit`** |
| `diff_method` | `best` | **`backprop`** |
| Interface | Torch | Torch |
| Shots | Analytic (`None`) | Analytic (`None`) |
| Qubits/depth | 4 / 3 | 4 / 3 |
| Trainable Q/K/V independence | Preserved | Preserved |
| Same CASE model/loss/batch in paired gate | Yes | Yes |

`best` is a method-selection setting, **not evidence of a particular differentiation algorithm for every original run**. Avoid claiming that it necessarily used parameter-shift, adjoint, or a specific number of circuit derivative evaluations without direct measurement.

### 5.2 Mechanistic explanation

With Torch-compatible backpropagation on `default.qubit`, the computation can form a differentiable statevector-based numerical graph, allowing reverse-mode differentiation through supported simulator operations. This can avoid expensive alternative gradient execution paths and repeated framework scheduling for small circuits. For the tested four-qubit and many-input workload, the combination proved much more effective than the earlier device/differentiation settings.

**Do not generalize** that `default.qubit` is universally faster than `lightning.qubit`: simulator advantages depend on model, interface, batching, differentiation, shots, data types and hardware. Also, the presence of a CUDA GPU does not alone certify dedicated GPU-native quantum simulation.

### 5.3 Paired real-CASE numerical and runtime gate (jobs 8255 and 8256)

Both candidates were run with the **actual CASE loss**, **real dataset batch**, **identical initial weights**, and **one optimizer step**. The tests assessed scalar loss, predictions, gradients and updated parameters.

| Model | Reference one-step wall time | Candidate one-step wall time | Reference/candidate ratio |
|---|---:|---:|---:|
| B3 | 8.126 s | **1.375 s** | **5.91×** |
| QSAL | 184.919 s | **0.511 s** | **361.9×** |

Numerical results:

- Identical scalar loss to the gate's reporting tolerance.
- Maximum absolute output discrepancy **1.67×10⁻⁶** (B3) and **2.15×10⁻⁶** (QSAL).
- Parameter-gradient differences below **7×10⁻⁷**; 291 B3 and 280 QSAL parameter-gradient tensors checked.
- Updated-parameter differences below **5.6×10⁻⁵**.
- Required numerical gates passed according to the experiment record.

**Correct claim:** one-step equivalence and acceleration *for those paired tests*. **Not demonstrated:** 361.9× lower steady-state QSAL training time, 361.9× full-epoch speedup, or multi-epoch equivalence. In fact, the candidate QSAL one-step 0.511 s is of the same order as its later ~0.58 s/step batch-8 benchmark, whereas the 184.919 s reference is unusually slow; the original experiment may carry substantial backend-specific overhead. Do not re-label it as a repeated median.

### 5.4 What part of the speedup belongs to which change?

The T=2 **7934→7937** paired study isolates the **batching implementation**, with fixed original backend. The **8255/8256** paired gates assess **the reference backend/differentiation path versus the new path in the tested implementation**. They are not a full factorial ablation across batch implementations, backend choices, GPU types and complete epochs. Thus, **do not multiply 1.452× by 361.9×**, or assign exact percentages of final wall-time savings to each method. The defensible conclusion is that both changes contribute to an efficient adopted pipeline, with different degrees of direct causal isolation.

---

## 6. Repeated candidate timing before the epoch run

### 6.1 Full-length, batch size 2 optimised short benchmarks

Jobs **8260 (B3)** and **8261 (QSAL)** used T=120, eight modalities, H=20, batch size 2, ten warm-up steps, fifty measured steps, three repetitions, FP32, and A100 2g.20gb MIG hardware.

| Model | Optimised seconds/step, repeated range | Approximate central observation |
|---|---:|---:|
| B3 | **0.0903–0.0909** | ~0.0905 |
| QSAL | **0.2355–0.2364** | ~0.2361 |

These are *candidate-only* short-run measurements; no matched repeated reference medians were completed in this same protocol. They are not epoch times.

### 6.2 Full-length, batch size 8 candidate short benchmark

Job **8262**, T=120, eight modalities, H=20, batch size 8, five warm-up steps, twenty measured steps, one timing repetition, A100 2g.20gb MIG:

| Model | Measured candidate seconds/step |
|---|---:|
| B3 | **0.10636** |
| QSAL | **0.58506** |

This short-run evidence was an appropriate *sanity check* for the later batch-8 epoch. It is not a rigorous spread estimate, and the full-A100 node used by job 8306 is different hardware.

### 6.3 Why not claim “original/optimised seconds per step” from mismatched jobs?

For illustration, using the separate B3 reference 8026 (~8.04 s/step, RTX A6000, batch 2) and optimised 8260 (~0.0905 s/step, A100 MIG, batch 2) yields a **cross-hardware observational ratio of about 88.8×**. **This is not an isolated optimisation speedup.** Hardware, software execution path, and benchmark phases differ, so it belongs only in a prominently qualified historical comparison, not as a central thesis result.

QSAL's original 8236 timed out, meaning there is no completed, matched 50-step reference timing series to compare to the new repeated candidate on equal terms.

---

## 7. Real full-epoch optimised training: SLURM job 8306

### 7.1 Exact experiment identity

| Field | Job 8306 evidence |
|---|---|
| Date | 21 November 2026 (Europe/Berlin) |
| Job/accounting | `COMPLETED`, exit `0:0`, **43m40s** total allocated walltime |
| Branch | `perf/b3-qsal-runtime-opt` |
| Git commit | `0425b741b5b78abbf513f6f586dd986ae7bc7aab` |
| Node | `zits-slurmworker1` |
| GPU | **NVIDIA A100 80GB PCIe**, full GPU; **not** MIG |
| Hardware metadata | GPU memory **79.277 GiB** (PyTorch-reported), Torch `2.1.0+cu121`, CUDA `12.1` |
| SLURM resources | 1 node, 1 task, **1** full A100 GPU, 4 CPUs, 32 GB requested memory, 3 h limit |
| Quantum | `default.qubit`, `backprop`, analytic shots, 4 qubits, depth 3 |
| Data | CASE: **28,560** train / **7,140** validation windows |
| Batches | Batch size **8**; **3,570** train batches and **893** validation batches |
| Temporal | `T=120`, history `K=120`, forecast horizon `H=20`, 2 Hz |
| Optimizer / scheduler | AdamW, LR `0.0003`, weight decay `1e-5`; ReduceLROnPlateau configured |
| Reproducibility | seed `42`, deterministic `True`; no AMP reported |
| Training cap | `epochs=1`, `max_train_batches=null`, `max_val_batches=null` |

Job 8306 sequentially launched the actual training entrypoints, B3 first, then QSAL, on the same allocated node/GPU. Times are *entrypoint wall times* measured by shell timestamps; they include per-model initialization, data index loading, real training, validation, checkpoint creation and finalization.

### 7.2 Measured performance and loss

| Quantity | B3 | QSAL |
|---|---:|---:|
| Training batches | 3,570 | 3,570 |
| Training progress duration | **6m40s (400 s)** | **34m36s (2,076 s)** |
| Training throughput (progress display) | **8.91 batches/s** | **1.72 batches/s** |
| Validation batches | 893 | 893 |
| Validation progress duration | **44 s** | **1m21s (81 s)** |
| **Full entrypoint measured** | **451 s = 7m31s** | **2,164 s = 36m04s** |
| Other entrypoint elapsed (by subtraction, approximate) | ~7 s | ~7 s |
| Epoch train loss | **0.5981** | **0.6742** |
| Epoch validation loss | **3.6885** | **3.7157** |
| Saved best/last checkpoints | Yes | Yes |

**Measured end-to-end cost ratio:** `2164 / 451 = 4.7982` (QSAL is about **4.80×** slower than B3). B3/QSAL are *different fusion architectures*, so this ratio is **not** an optimisation speedup ratio.

**Timings are not additive to 1-second precision:** progress-bar durations are rounded display figures, entrypoint times are integer-second shell intervals, and job-walltime includes inter-model/setup/reporting work. Job 8306's **43m40s** SLURM elapsed is consistent with, but not exactly the same measurement scope as, the **43m35s** sum of model entrypoint seconds.

### 7.3 Quality results: valid finite training, not proven generalization

Both models returned finite one-epoch loss values and saved checkpoints. However, validation losses (3.6885/3.7157) substantially exceed train losses (0.5981/0.6742). This is a **quality/evaluation question**, separate from runtime engineering. A one-epoch difference does not by itself prove overfitting; subject-independent distribution shift, normalization, loss aggregation, label range, train/eval semantics and metric implementation all merit checking.

QSAL's `history.json` stores 893 per-validation-batch metric objects; many per-batch CCC/PCC values are zero or unstable. The provided `training/trainer.py` excerpt appends `evaluate_outputs(...)` metrics once per validation batch. Before claiming dataset-level CCC/PCC, verify how constant/low-variance batches are handled and evaluate predictions at a well-defined pooled/subject-level unit, avoiding an unjustified unweighted mean of per-batch correlations. Neither one epoch nor a checkpoint filename establishes strong forecasting accuracy.

### 7.4 Model size and checkpoint evidence

| Model | Recorded trainable parameter count | Parameter memory estimate |
|---|---:|---:|
| B3 | **8,073,056** | ~30.80 MB |
| QSAL | **8,121,553** | ~30.98 MB |

Actual training run paths (the training framework appends its own timestamp):

```text
runs/case_b3_a10080_epoch1_retry2/20261121_154621/
    config_used.json
    params.json
    best.pt
    last.pt
    history.json
    metrics.csv

runs/case_qsal_a10080_epoch1_retry2/20261121_155353/
    config_used.json
    params.json
    best.pt
    last.pt
    history.json
    metrics.csv
```

**Do not confuse** a configured `logging.run_dir` (parent) with the actual timestamped child directory created by `training/base_training_script.py`.

### 7.5 Direct machine-readable evidence

```text
benchmark_results/case_a10080_epoch1_final/job_8306/
    epoch_duration.csv
    epoch_duration.json
    hardware.json
    gpu_information.txt
    case_b3_a10080_epoch1_retry2.yaml
    case_qsal_a10080_epoch1_retry2.yaml

slurm/case_a10080_epoch1_final_8306.out
slurm/case_a10080_epoch1_final_8306.err
slurm/case_a10080_epoch1_single_gpu_fixed.sbatch

configs/experiments/case_b3_a10080_epoch1_retry2.yaml
configs/experiments/case_qsal_a10080_epoch1_retry2.yaml
```

The experiment JSON stores the actual GPU name, Torch/CUDA versions, dataset window counts, batch size, backend/differentiation selection, complete-entrypoint timing scope and measured elapsed seconds.

### 7.6 Planning a larger training run — estimates only

**Naïve 30× single-epoch entrypoint extrapolation:** B3 `30 × 451 = 13,530 s = 3h45m30s`; QSAL `30 × 2164 = 64,920 s = 18h02m00s`. Combined sequential: **21h47m30s**. This overcounts repeated per-process startup if all 30 epochs run within one invocation, and it does not model later-epoch dynamics, checkpoint effects, or early stopping. It is **not a measured 30-epoch result**, nor a guaranteed SLURM runtime.

Separate full-run submissions, adequate walltime margin, scheduler-policy review, per-epoch intermediate checkpoints and tested cross-job resume logic are required before spending long allocations. The training entrypoint's timestamped run-directory behavior means one cannot assume a new SLURM job automatically resumes from an old `last.pt` checkpoint.

---

## 8. Submission and failure chronology: provenance, not benchmark data

The earlier launch failures were administrative and **must not be included in runtime-speed calculations**.

| Job | Outcome | Relevance |
|---|---|---|
| 8302 | Canceled while pending RTX A6000 | A6000 wait avoided; no epoch measurement |
| 8303 | Canceled after ~6m38s | `--nodelist=zits-slurmworker[1-3]` unintentionally allocated **3 nodes / 3 GPUs**; B3 initialization began, no full epoch |
| 8304 | Failed in ~4 s | Correct 1-node/1-GPU allocation; missing `git` on compute node and run-directory collision |
| 8305 | Failed before training | Compute-node Git call removed but preflight still referenced original epoch1 YAML, causing stale run-directory rejection |
| **8306** | **COMPLETED, exit 0** | Final launcher preflight points to the `retry2` YAMLs; **real B3 and QSAL epochs completed** |

The full-A100 scheduling correction was `--nodes=1`, `--ntasks=1`, typed `--gres=gpu:nvidia_a100_80gb_pcie:1` and exclusion of unsuitable login/MIG nodes; it is a **resource-allocation fix**, not an algorithmic optimisation. All historical logs, partial artifacts and scripts should remain intact.

---

## 9. A fair and thesis-defensible comparison framework

### 9.1 Three evidence tiers — never merge the ratios

| Tier | Evidence | Supported conclusion | Unsupported interpretation |
|---|---|---|---|
| **A — Batching-only microbenchmark** | Jobs 7934/7937; same original simulator; T=2 | QNode dispatch reduced; B3 1.317×, QSAL 1.452× faster forward median | Full-length or full-epoch speedup |
| **B — Paired backend gate** | Jobs 8255/8256; real batch, one optimizer step | Outputs/gradients/updated weights agree; B3 5.91×, QSAL ~362× in that step | Steady-state 362× training acceleration |
| **C — Operational candidate** | Jobs 8260/8261/8262 (short); **8306 complete epoch** | Candidate can train full CASE epochs on a full A100; exact runtimes | Isolated reference-vs-candidate full-epoch gain |

### 9.2 Optimised versus non-optimised: what can be compared numerically?

| Comparison | Reference | Optimised | Validity |
|---|---:|---:|---|
| Batching-only B3 forward, T=2 | 89.054 ms | 67.642 ms | **Matched bounded microbenchmark** |
| Batching-only QSAL forward, T=2 | 1717.533 ms | 1183.204 ms | **Matched bounded microbenchmark** |
| B3 paired one-step backend gate | 8.126 s | 1.375 s | **Paired one-step**; not repeated epoch |
| QSAL paired one-step backend gate | 184.919 s | 0.511 s | **Paired one-step**; likely substantial overhead sensitivity |
| B3 historical 8026 vs candidate 8260 | ~8.04 s/step, RTX A6000 | ~0.0905 s/step, A100 MIG | **Cross-hardware historical observation only** |
| QSAL historical 8236 vs candidate 8261 | Reference timed out | ~0.2361 s/step, A100 MIG | **No numerical ratio justified** |
| Historic report vs job 8306 | Claimed 15.02/31.81 h projected epoch, arithmetic inconsistent | 7m31s/36m04s **measured** | **Illustrates improved feasibility, not matched causal epoch speedup** |

### 9.3 Requirements for a future matched full-epoch comparison

If time/resources permit a formal reference/candidate experiment, freeze the same CASE manifest, source propagation, code revision except runtime implementation, T/K/H, qubits/depth, seed, optimizer, precision, PyTorch/PennyLane versions, batch size, GPU SKU and allocation, CPU resources, data workers, timing scope, and cache state. Confirm the reference is semantically correct for eight-modality QSAL. Record startup/train/validation/checkpoint time separately; synchronize CUDA timing regions; run multiple repetitions or clearly report censored timeouts. Do **not** incur another 2-hour QSAL original reference timeout merely to manufacture a number. If a matched baseline is infeasible, report the incomplete evidence as a limitation.

---

## 10. Mechanistic interpretation and remaining computational cost

B3 evaluates a unified PQC after aggregation. QSAL produces three quantum attention components over individual modality tokens, followed by Gaussian attention, and must differentiate through their respective quantum branches during training. Batching reduces Python orchestration and a better-differentiated simulator changes autograd cost, but the QSAL architecture still performs more logical quantum work.

The 8306 progress durations confirm a larger disparity in training (QSAL **2076/400 = 5.19×** B3) than validation (**81/44 = 1.84×**). This is *consistent* with additional QSAL gradient work; a detailed forward/backward profiler is still needed to quantify the actual cost attributable to the quantum backward pass. Note also that B3 and QSAL loss values alone are **not** evidence of better quality without final evaluation after equivalent full training.

---

## 11. Future optimisation roadmap — ranked and gated

**Priority principle:** Do not jeopardize the verified 8306 baseline without a clear measured bottleneck and controlled correctness gate.

| Priority | Candidate | Hypothesis | Required gate / trade-off | Implemented in 8306? |
|---|---|---|---|---|
| 1 | Isolate quantum forward/backward and host-dispatch time at **T=120** | Identify whether QSAL still spends most time in its Q/K/V gradients | Profile one representative real batch; synchronized GPU/CPU timing; preserve checkpoints | No — future profiling |
| 2 | Evaluate Q/K/V batching scheduling or execution chunking while keeping separate parameters | Reduce orchestration/memory peaks on long sequences | Exact outputs/gradients/updates and peak memory; compare 3+ repeated times | No |
| 3 | Profile host-device transfers and DataLoader workers | Remove unnecessary stalls | Show transfer/loader share before changing; avoid silent dtype changes | No |
| 4 | Evaluate exact alternative differentiable backends | Reduce simulation/autograd overhead further | Supported Torch gradients; exact gate/measurement semantics; paired correctness | Current backend adopted; alternatives deferred |
| 5 | Consider selective AMP for classical layers only | Classical transformer acceleration if dominant | Separate precision ablation and stability checks; never assume PQC AMP support | No |
| 6 | Multi-GPU parallelism, only with scaling evidence | Greater throughput for sufficiently large workloads | One-vs-two GPU scaling and resource efficiency; no duplicate 3-node allocation | No |
| 7 | Caching invariant projections / avoiding duplicate equivalent work | Reduce nonsemantic repeated work | Demonstrate what is invariant over optimizer steps; keep autograd intact | Not demonstrated |

**Not an implementation-only speedup:** reducing 4 qubits to 2, depth 3 to 1, eight to one modality, using shorter T/H, sharing Q/K/V parameters, removing Gaussian attention, or reducing training batches/epochs. Those belong to explicitly controlled architecture or data ablation studies.

**Stop criterion:** Once the full-length model is operational and the remaining cost does not obstruct scientific experiments, prioritize data integrity, evaluation, model quality and final controlled comparisons over increasingly speculative micro-optimisations.

---

## 12. Reproducibility runbook

### 12.1 Working directory and prerequisites

```bash
conda activate hybrid-qemotion
cd "$HOME/projects/Hybrid-Quantum-Classical-Multimodal-Continuous-Emotion-Recognition-and-Trajectory-Forecasting-b3-qsal-opt"

# Read-only identity checks
pwd
git rev-parse HEAD
git status --short
python -c 'import torch, pennylane as qml; print("torch", torch.__version__, "pennylane", qml.__version__)'
```

Only use the `hybrid-qemotion` environment and the correct worktree. A commit hash does not capture untracked scripts and YAML files; maintain their own copies/hashes for archival. The 8306 node could not find a Git executable, so the final script embedded the commit captured on the login node instead of invoking Git from the compute node.

### 12.2 Final launcher and config paths

```text
slurm/case_a10080_epoch1_single_gpu_fixed.sbatch
configs/experiments/case_b3_a10080_epoch1_retry2.yaml
configs/experiments/case_qsal_a10080_epoch1_retry2.yaml
training/train_case_b3.py
training/train_case_qsal.py
training/base_training_script.py
training/trainer.py
```

The **final** SLURM script is the `single_gpu_fixed.sbatch` file. `single_gpu_retry2.sbatch` is an earlier intermediate script with an incorrect preflight YAML reference and should not be submitted as the final verified protocol.

### 12.3 Archive the exact submitted script

The final script is already on disk. To capture Slurm's stored copy, **if not already captured**, on the HPC login node:

```bash
# Do not overwrite an existing archive.
if [ ! -e slurm/case_a10080_epoch1_8306_submitted.sbatch ]; then
  scontrol write batch_script 8306 slurm/case_a10080_epoch1_8306_submitted.sbatch
fi
```

If the scheduler has purged this historic job's script, retain the preexisting final launcher and recorded job logs; do not claim a recovered scheduler copy exists if it does not.

### 12.4 Inspect the completed run without launching another job

```bash
sacct -j 8306 --format=JobID,JobName,State,Elapsed,ExitCode,NodeList
cat benchmark_results/case_a10080_epoch1_final/job_8306/epoch_duration.json
cat benchmark_results/case_a10080_epoch1_final/job_8306/epoch_duration.csv

grep -E '\[Epoch 1\]|entrypoint elapsed|ALL MEASUREMENTS COMPLETED' \
  slurm/case_a10080_epoch1_final_8306.out

find runs/case_b3_a10080_epoch1_retry2 \
     runs/case_qsal_a10080_epoch1_retry2 \
  -maxdepth 3 -type f -print
```

The `.err` file may contain tqdm training progress lines, not Python errors; read both streams and check `sacct` exit status.

### 12.5 Safe rerun protocol (do not blindly resubmit)

**Do not simply rerun** `sbatch slurm/case_a10080_epoch1_single_gpu_fixed.sbatch`: the script protects existing output directories and may reject an already-used configured run root. For a new experiment:

1. Copy both YAMLs to **new filenames**, preserving every scientific setting and changing only `logging.run_dir` to **never-before-used** paths.
2. Copy the final `.sbatch` to a **new script filename**; update **both training command references and the preflight `Path(f"...{model}...")` template**, all three references in total, to the new YAML filenames.
3. Ensure a unique benchmark `RESULTS` root per job, separate stdout/stderr file names or `%j`-resolved names, and a fresh job name if useful.
4. Run `bash -n` on the copied launcher, verify its `#SBATCH --nodes=1`, `--ntasks=1`, `--gres=gpu:nvidia_a100_80gb_pcie:1`, and inspect config paths with `grep`.
5. Confirm neither new `logging.run_dir` exists **before** submission. A training entrypoint additionally creates a timestamped child directory.
6. Run `sbatch <new-script>` only after the above preflight. Check scheduler allocation, startup logs, training/validation completion, checkpoint existence and reporting outputs.

Do not delete old run roots, overwrite job 8306's archived configs/logs, or let another worktree reuse their paths. For full multi-epoch runs, adjust epochs and SLURM walltime **in separately named configurations** and validate restart behavior independently.

### 12.6 Preserve a compact reproducibility bundle

Recommended records (do not pack large `.pt` files into Git):

```text
runtime_evidence/
  README.md                      # This report or a pointer to it
  slurm_8306.sbatch              # Exact stored submission if retrievable
  b3_config_used.json
  qsal_config_used.json
  b3_params.json
  qsal_params.json
  epoch_duration.csv
  epoch_duration.json
  hardware.json
  sacct_8306.txt
  SHA256SUMS
```

Create an archive by **copying** from source files, never moving or deleting originals, and log hashes with `sha256sum` where available. Keep `best.pt`, `last.pt`, `history.json`, `metrics.csv` and the complete SLURM `.out`/`.err` at their original retained paths or in an independent immutable backup. Large data and checkpoints may require storage beyond the Git repository.

---

## 13. Evidence registry and provenance

| Claim | Strongest available evidence / path | Status |
|---|---|---|
| Historical 15.148/32.080 s/batch and reported 15.02/31.81 h/epoch | `docs/phases/phase1_case_completion_report.md`, computational benchmark | Historical; protocol arithmetic unresolved |
| Source-propagation defect in QSAL and correction | Existing technical record §§18–19; job 7919, source tests | Verified correction; invalidates 7898 as eight-source baseline |
| Original serial → batched QNode invocation counts | Technical record §20; `artifacts/runtime_opt/task2_20261117_2154/`; jobs 7934/7937 | Verified T=2 call-count reduction |
| Batching-only forward and backward timing | Same profiler jobs and JSON | Bounded synthetic T=2, three forward medians, single F+B |
| Batched-output and gradient agreement | Same profiler equivalence artifacts | Verified T=2 test |
| Simulator/diff one-step paired correctness | `b3_backend_pair_8255.out`, `qsal_backend_pair_8256.out` and saved gate artifacts | Verified for one real optimizer step |
| Candidate batch-2 short performance | Jobs 8260/8261 outputs | Three repeats, 10 warm-up, 50 measured |
| Candidate batch-8 short performance | Job 8262 output | One repeat, 5 warm-up, 20 measured |
| Final one-epoch runtime | `benchmark_results/case_a10080_epoch1_final/job_8306/epoch_duration.{json,csv}` | Complete real epoch, one full A100 |
| Successful scheduler status | `sacct -j 8306`, `.out` and `.err` | `COMPLETED`, exit `0:0` |
| B3 losses / checkpoints | `runs/case_b3_a10080_epoch1_retry2/20261121_154621/` | One epoch, saved |
| QSAL losses / checkpoints | `runs/case_qsal_a10080_epoch1_retry2/20261121_155353/` | One epoch, saved |
| Historic plans and guardrails | `docs/B3_QSAL_Runtime_Optimisation_Technical_Record.md` | Retain as living historical record |

**Evidence boundary:** The original HPC repository was inspected by an SSH-enabled assistant, whereas this report has been assembled from the supplied technical record, original Phase 1 source, user-provided 8306 output, and its inspection summary. The present document does **not** claim to have independently executed HPC commands, fetched inaccessible jobs, or confirmed that a proposed archive currently exists.

---

## 14. Thesis discussion: validity, limitations and significance

### 14.1 Internal validity

The batching study controls the simulator and uses numerical checks, providing direct evidence that dispatch vectorisation reduces overhead. The backend study uses paired real CASE data and identical starting states to establish one-step numerical agreement. Together these support the engineering rationale, but there is no complete factorial timing experiment isolating the individual changes under the same 8306 hardware and one-epoch protocol.

### 14.2 Construct validity of runtime

A forward-only microbenchmark, a forward+backward single step, a training progress bar and a complete training entrypoint measure different quantities. No one ratio should substitute for another. Training entrypoint time includes actual optimizer updates, validation and artifact writing, making it the best resource-planning measure of the completed candidate but not automatically a causal improvement statistic.

### 14.3 External validity and hardware

The microbenchmarks used an A100 MIG slice and the earlier B3 reference used an RTX A6000; job 8306 used a full A100 80GB PCIe. These GPUs differ, and a full A100 is not guaranteed to accelerate every CPU-dispatch-dominated quantum workload. Generalization to other PennyLane versions, CUDA configurations, circuits, qubit counts or commercial QPUs is unsupported.

### 14.4 Numerical versus scientific equivalence

Output/gradient/optimizer agreement for fixed inputs supports **one-step functional equivalence** under the selected tolerances. It does not establish bitwise determinism or matching final emotion-prediction accuracy after 30 epochs and multiple seeds. Report training outcomes and statistical uncertainty separately.

### 14.5 Dataset and metric quality

The intended CASE split is subject independent. One-epoch validation loss is far above training loss for both architectures, so any thesis claims of predictive benefit must rely on separate, properly aggregated evaluation after adequate training. Avoid treating per-batch CCC/PCC values as global metrics without handling near-constant prediction/target variance, subject weighting and temporal overlap.

### 14.6 Computational significance

The practical contribution is the elimination of an operational training bottleneck while preserving experimental design. The post-optimisation models finished complete epochs with usable checkpoints, enabling subsequent long-run and architecture-controlled emotion forecasting studies. The strong paired short-step results are encouraging; the limitation is that no trustworthy hardware- and protocol-matched complete reference epoch is available.

---

## 15. Thesis-ready text (adapt to thesis structure)

### 15.1 Problem and intervention

> Preliminary computational profiling identified the differentiable parameterised quantum circuit (PQC) execution path as a dominant bottleneck in the B3 unified quantum fusion and QSAL quantum self-attention architectures. The original implementation issued one QNode call per flattened input sample, multiplying Python-to-simulator dispatch and gradient-computation overhead along the temporal and modality axes. An implementation-preserving optimisation was introduced by broadcasting a batch of angle-encoded PQC inputs through each existing QNode. A further backend intervention selected PennyLane's `default.qubit` simulator with Torch-compatible backpropagation rather than the historical `lightning.qubit` configuration with automatic differentiation-method selection. The number of qubits, circuit depth, parameterization, input modalities, attention semantics, prediction heads and temporal configuration were kept unchanged.

### 15.2 Validation of changes

> On a bounded T=2 profile with the original backend fixed, Python-level QNode invocations fell from 16 to one for B3 and from 384 to three for QSAL, while measured median forward time improved by factors of 1.317 and 1.452, respectively. Numerical checks verified output and gradient agreement for that batching change. Subsequent paired tests on real CASE batches and identical model states compared the reference and adopted differentiation paths, obtaining one-step wall-time ratios of 5.91 for B3 and approximately 362 for QSAL; prediction, gradient and parameter-update discrepancies remained within the recorded numerical gates. These step-level ratios are not interpreted as full-epoch speedup factors.

### 15.3 Full-epoch operational results

> The final optimised implementations were assessed in SLURM job 8306 on one NVIDIA A100 80GB PCIe GPU with batch size eight and the full CASE training and validation partitions (28,560 and 7,140 windows, respectively). The configuration used eight physiological modalities, a 120-step observation/history sequence and a 20-step forecasting horizon. The end-to-end entrypoints, including initialization, training, validation, checkpointing and finalization, required 451 seconds for B3 and 2,164 seconds for QSAL. Both completed successfully, saved checkpoints and produced finite losses. QSAL remained 4.80 times more computationally expensive than B3 at this operating point. The historical per-batch baseline and its projected epoch durations are retained as motivation but not used to claim a matched end-to-end acceleration because the original protocol and hardware differ, the hour projections contain a batch-denominator inconsistency, and at least one early QSAL reference used incorrect modality routing.

### 15.4 Limitations and continuation

> The reported evidence demonstrates implementation correctness at the tested step and operational feasibility of the full-length optimised models. It does not establish complete reference-to-candidate epoch acceleration under matched hardware or equivalent learning outcomes after long training. Future work should prioritise a full-length quantum backward profiler, validated chunking or scheduling of independent Q/K/V evaluations, and robust dataset-level recognition/forecasting evaluation. Architecture-changing shortcuts must be assessed as separate ablation experiments rather than included in performance-preserving optimisation claims.

---

## 16. Recommended document maintenance policy

**Keep** `docs/B3_QSAL_Runtime_Optimisation_Technical_Record.md` (the original, historically ordered technical runbook) **unchanged** to preserve provenance. **Add this file** in the same directory under the name `B3_QSAL_Runtime_Optimisation_Results_and_Thesis_Report.md`. Optionally append a short dated cross-link at the end of the original document, clearly labelled **“Post-implementation results index (2026-11-22)”**; do not erase prior entries, rewrite historical claims silently or replace source logs. Going forward:

- Update this companion only when a new **verified** result or material correction exists, with job ID, commit, protocol and evidence path.
- Keep exact SLURM scripts, configurations, JSON/CSV and checkpoints linked to their immutable experiment IDs.
- Put speculative or untested optimisations in the future-work table, not in “achieved results”.
- Make the eventual thesis chapter cite the completed run and method-specific correctness gates, not an unsupported single overall speedup number.

**Current final status:** **B3 and QSAL optimised one-epoch feasibility proven; precise matched non-optimised versus optimised full-epoch acceleration still unmeasured.**

---

## 17. Master experiment ledger — job IDs, evidence and admissible claims

Use this table as the **index of experimental evidence**, not as a table of mutually comparable speed observations. Runs differ in model correctness, GPU, batch size, temporal length, warm-up protocol and timing scope. Dates come from existing records where available; do not infer an exact date solely from ascending job IDs.

| Job ID | Experimental role / variant | Hardware and workload | Outcome / numeric result | What may be claimed |
|---|---|---|---|---|
| **7742** | Initial manual baseline log capture | Early optimisation-stage allocation | Archived `b3_case_gpu.log` and `training_speed_gpu.log` | Initial state capture only; not a full-epoch result |
| **7898** | Early QSAL reference before source propagation correction | QSAL source wiring selected only ECG, despite eight enabled CASE modalities | Historical, scientifically **ineligible** as an eight-source QSAL baseline | Preserve as a defect record; **exclude** from corrected-QSAL runtime ratio |
| **7919** | Source-propagation validation | Real CASE, batch 2, T=120; no backward/optimizer | Corrected eight-source QSAL output verified `(2,120,8,128)` source stack | Confirms corrected modality routing; **not** a timing baseline |
| **7930** | T=2 profiling harness attempt | Bounded synthetic inputs | Failed due to unsupported QNode `shots` attribute access | Harness debugging only |
| **7932** | T=2 profiling harness attempt | Bounded synthetic inputs | B3 forwards ran; backward failed because GRU was in eval mode | Not a valid complete comparison |
| **7934** | **Before batching**, `lightning.qubit` + `best` | A100 MIG 2g.20GB; synthetic T=2, B=8, 8 sources | B3 median forward **89.054 ms**; QSAL **1717.533 ms** | Batching-control baseline (matched to 7937) |
| **7935** | T=2 post-batching harness attempt | Bounded synthetic inputs | Stale expected call count caused failure before complete QSAL profiling | Retain as failed attempt, not final measurement |
| **7937** | **After batching**, same original backend | Same profile shape and hardware class as 7934 | B3 **67.642 ms**; QSAL **1183.204 ms**; QNode calls **1/3** | Isolated bounded **forward** batching benefit: B3 **1.317×**, QSAL **1.452×** |
| **8026** | B3 historical, unoptimised reference step benchmark | RTX A6000 48GB; batch 2; 10 warm-ups, 50 timed, three repeats | Approximately **8.04 s/step** | Historical B3 step-scale burden; **not** same-hardware as job 8306 |
| **8236** | QSAL reference timing attempt | A100 MIG 2g.20GB; original reference backend | **Timed out after 2 h**; incomplete measured series | Reference is expensive/censored; **no invented QSAL step or epoch time** |
| **8255** | B3 paired real-CASE backend/correctness gate | Same input and initial parameters within paired comparison; one optimizer step | **8.126 s** reference vs **1.375 s** candidate, **5.91×** | Paired one-step acceleration and numerical gate, **not epoch** |
| **8256** | QSAL paired real-CASE backend/correctness gate | Same input and initial parameters within paired comparison; one optimizer step | **184.919 s** reference vs **0.511 s** candidate, **361.9×** | Paired one-step acceleration and numerical gate, **not sustained epoch** |
| **8260** | Repeated optimised B3 batch-2 short benchmark | A100 MIG 2g.20GB; T=120; 10 warm-ups, 50 measurements × 3 | **0.0903–0.0909 s/step**, approx. **0.0905** | Steady short-benchmark candidate throughput, **not full epoch** |
| **8261** | Repeated optimised QSAL batch-2 short benchmark | A100 MIG 2g.20GB; T=120; 10 warm-ups, 50 measurements × 3 | **0.2355–0.2364 s/step**, approx. **0.2361** | Steady short-benchmark candidate throughput, **not full epoch** |
| **8262** | Optimised batch-8 short comparative benchmark | A100 MIG 2g.20GB; T=120; 5 warm-ups, 20 measurements × 1 | B3 **0.10636 s/step**, QSAL **0.58506 s/step** | Candidate scaling observation; single repetition |
| **8302** | Preliminary RTX A6000 one-epoch attempt | Awaiting allocation | Cancelled before benchmark | No measured training time |
| **8303** | Preliminary full-A100 one-epoch launcher | Faulty `--nodelist` allocated **3 nodes / 3 GPUs**; training not completed | Cancelled after approx. 6m38s | Scheduler-allocation mistake, not model speed data |
| **8304** | Repaired single-GPU launcher, first attempt | Full A100 80GB worker1, one node | Failed after ~4s; compute node lacked `git` and preflight found stale run directory | Correct resource allocation; no training benchmark |
| **8305** | Second single-GPU launcher | Full A100 80GB worker1, one node | Failed pre-training; preflight read old epoch1 YAML | No training benchmark |
| **8306** | **Final optimised B3 and QSAL full-epoch run** | **Full A100 80GB**, B=8, T=120, K=120, H=20, 8 sources, 28,560 train/7,140 validation windows | **COMPLETED, exit 0:0**; B3 **451 s**, QSAL **2164 s**; job elapsed **43m40s** | Actual complete *optimised* one-epoch entrypoint times; not a matched legacy-versus-optimised epoch speedup |

**T=2 batching artifact directory:** `artifacts/runtime_opt/task2_20261117_2154/` (before/post JSON, profiling harness, output and errors). **Paired backend artifacts:** `b3_backend_pair_8255.out`, `qsal_backend_pair_8256.out`, plus gate artifacts referenced by the technical record. **8306 primary evidence:** `benchmark_results/case_a10080_epoch1_final/job_8306/epoch_duration.json` and `slurm/case_a10080_epoch1_final_8306.out`.

## 18. Numeric comparison register — exact meanings and calculations

### 18.1 Non-optimised versus optimised: comparisons that control the changed method

| Model | Controlled question | Non-optimised observation | Optimised observation | Absolute time saved | Ratio | Evidence level |
|---|---|---:|---:|---:|---:|---|
| B3 | **Batching only**, T=2 forward median | 89.054 ms | 67.642 ms | 21.412 ms | **1.317×** | Profiling jobs **7934/7937** |
| QSAL | **Batching only**, T=2 forward median | 1717.533 ms | 1183.204 ms | 534.329 ms | **1.452×** | Profiling jobs **7934/7937** |
| B3 | Backend/differentiation paired one optimizer step | 8.126 s | 1.375 s | 6.751 s | **5.91×** | Correctness/performance job **8255** |
| QSAL | Backend/differentiation paired one optimizer step | 184.919 s | 0.511 s | 184.408 s | **361.9×** | Correctness/performance job **8256** |

**Definition:** `speed ratio = reference duration / candidate duration`; `time reduction (%) = 100 × (1 − candidate/reference)`. In those **specific measured scopes**, batching-only forward time reductions were **24.0% B3** and **31.1% QSAL**; paired one-step reductions were **83.1% B3** and **99.7% QSAL**. These percentages **must not** be relabelled as full-epoch savings or multiplied together.

The batching-only measurement also records **one forward+backward sample** (not a median): B3 **1.323808 → 0.848950 s** (approximately 1.559×); QSAL **20.589187 → 19.653934 s** (approximately 1.048×). The relatively modest QSAL improvement in that sample is consistent with a remaining differentiation bottleneck on the historical backend and motivated the backend screen, but it does not by itself prove causality.

### 18.2 Historical Phase 1 epoch projections: denominator audit

Let `N_train = 28,560` windows and `n_batches = ceil(N_train / batch_size)`. An estimate using per-training-batch time is `estimated_train_time = seconds_per_batch × n_batches`. It is **not** an end-to-end epoch measurement: validation, startup, checkpointing and other overhead are excluded unless explicitly included in the timing sample.

| Historical quantity | At batch size **2** (14,280 batches) | At batch size **8** (3,570 batches) | Original report's published epoch estimate |
|---|---:|---:|---:|
| B3, **15.148 s/batch** | **60h05m13s** | **15h01m18s** | **15.02 h** |
| QSAL, **32.080 s/batch** | **127h15m02s** | **31h48m46s** | **31.81 h** |

These computed columns are **hypothetical arithmetic using the stated per-batch rates**. The published hour estimates align with the batch-8 column, **not** with the report's stated batch-2 benchmark protocol. The true intended denominator and timing boundary were not recovered; preserve the original historical figures with this caveat.

### 18.3 Final real-epoch candidate results and repeatable planning quantities

| Quantity | B3 (job **8306**) | QSAL (job **8306**) | Meaning |
|---|---:|---:|---|
| Training, progress-bar reported | **400 s** (6m40s) | **2076 s** (34m36s) | Rounded train-phase duration |
| Validation, progress-bar reported | **44 s** | **81 s** (1m21s) | Rounded validation duration |
| Total training entrypoint | **451 s** (7m31s) | **2164 s** (36m04s) | **Measured**, includes initialization/finalization and checkpoint |
| Entrypoint minus displayed train+val | **~7 s** | **~7 s** | Approximate overhead by subtraction; progress bars rounded |
| Training batches | **3570** | **3570** | 28,560/8 |
| Validation batches | **893** | **893** | `ceil(7140/8)` |
| Training throughput (observed) | **8.91 batches/s** | **1.72 batches/s** | Progress bar; time-varying observation |
| Epoch training loss | **0.5981** | **0.6742** | One epoch; not final quality |
| Epoch validation loss | **3.6885** | **3.7157** | One epoch; subject-independent split |
| 30 × one-epoch entrypoint | **13,530 s = 3h45m30s** | **64,920 s = 18h02m00s** | **Extrapolation only**, not 30-epoch measurement |

**Combined measured entrypoint time:** `451 + 2164 = 2615 s = 43m35s`. **Combined SLURM elapsed:** `43m40s = 2620 s`; the approximately **5 s** difference is job wrapper/scheduler-related elapsed time at this reporting resolution. **QSAL/B3 entrypoint ratio:** `2164/451 = 4.7982×`.

The 30× projections count one-time startup repeatedly, whereas uninterrupted 30-epoch training ordinarily pays startup once. Training variability, checkpointing, validation, stopping rules and possible failures/resumes mean the true duration cannot be inferred exactly from one epoch. Do not quote a precise 30-epoch end-to-end acceleration against the unmatched old reference.

### 18.4 Why the original-reference and final-epoch numbers should never be joined into one speedup

| Confounder | Original historical or intermediate evidence | Job 8306 | Implication |
|---|---|---|---|
| Batch size | Historical Phase 1: labelled B=2 | **B=8** | Different steps/epoch and model/device load |
| GPU | Reference B3 8026: RTX A6000; QSAL 8236: A100 MIG 2g.20GB | **Full A100 80GB** | GPU capacity and available compute differ |
| Timing scope | Per-batch short/failed benchmark, projection | Complete entrypoint including validation/checkpoints | Numerator and denominator represent different workloads |
| QSAL source integrity | Early job 7898: only ECG routed | Eight sources validated | Old QSAL result not semantically comparable |
| Quantum execution | Serial QNode path and legacy backend | Batched `UnifiedPQC` + `default.qubit`/`backprop` | Multiple changes cannot be assigned one isolated effect |
| Repetitions | Unequal; some single timings or timeout | One complete epoch per model | Do not assign confidence intervals without repetitions |

## 19. Immutable artifacts and exact rerun decision table

### 19.1 Direct paths for completed job 8306

| Artifact | Relative path (from worktree root) | Role |
|---|---|---|
| **Final submitted launcher source** | `slurm/case_a10080_epoch1_single_gpu_fixed.sbatch` | Correct SLURM directives, preflight and sequential models |
| Optional SLURM-recovered submission | `slurm/case_a10080_epoch1_8306_submitted.sbatch` | Create with `scontrol write batch_script 8306 ...` only if retrievable and not already present |
| B3 config | `configs/experiments/case_b3_a10080_epoch1_retry2.yaml` | Full B3 training settings; `epochs=1` |
| QSAL config | `configs/experiments/case_qsal_a10080_epoch1_retry2.yaml` | Full QSAL training settings; `epochs=1` |
| B3 training entrypoint | `training/train_case_b3.py` | Existing production model entrypoint |
| QSAL training entrypoint | `training/train_case_qsal.py` | Existing production model entrypoint |
| Shared quantum implementation | `quantum/unified_pqc.py` | Broadcasting code under test |
| B3 run folder | `runs/case_b3_a10080_epoch1_retry2/20261121_154621/` | `best.pt`, `last.pt`, history, metrics, params, config |
| QSAL run folder | `runs/case_qsal_a10080_epoch1_retry2/20261121_155353/` | `best.pt`, `last.pt`, history, metrics, params, config |
| Timings | `benchmark_results/case_a10080_epoch1_final/job_8306/epoch_duration.csv` | B3=451 s, QSAL=2164 s |
| Detailed timing metadata | `benchmark_results/case_a10080_epoch1_final/job_8306/epoch_duration.json` | Job, node, device, CUDA and timing scope |
| Hardware metadata | `benchmark_results/case_a10080_epoch1_final/job_8306/hardware.json` | A100 GPU and software versions |
| Standard output | `slurm/case_a10080_epoch1_final_8306.out` | Train/validation completion, losses, and timestamps |
| Standard error/progress | `slurm/case_a10080_epoch1_final_8306.err` | tqdm progress; errors should be distinguished from progress text |

**Important:** The final `*_fixed.sbatch` differs from the earlier `*_retry2.sbatch`: the preflight now reads `case_{model}_a10080_epoch1_retry2.yaml`, **not** the obsolete epoch1 YAML. Also note the node lacked `git`; the final launcher captured and embedded the commit on the login node.

### 19.2 Rerun without overwriting scientific results

| Goal | Safe action | Do **not** do |
|---|---|---|
| Inspect or cite 8306 | Read SLURM `.out`/`.err`, `epoch_duration.*`, history and configs | Resubmit an existing run to fetch values already saved |
| Repeat one-epoch candidate for uncertainty | New paired YAML names, new `logging.run_dir`, new copied SLURM script, preserve same scientific settings | Reuse `retry2` run roots or overwrite earlier checkpoints |
| Train for 30 epochs | Create separately named 30-epoch configs; verify resume semantics and walltime first | Change existing 8306 configs or assume resume is automatic |
| Measure causal reference→candidate epoch acceleration | Matched hardware and batch, corrected eight-source QSAL, same timing scope; consider cost of timeout | Use 7898 or cross-hardware 8026/8306 as if controlled |
| Record published evidence | Save job IDs, Git commit, configs, SLURM directives, raw logs and tool versions | Cite a projection as measured, or a first-step ratio as a full-epoch speedup |

## 20. Thesis figures, tables and claim-to-evidence checklist

A thesis chapter can be built from the following data **without inventing missing performance values**.

| Thesis item | Data to present | Precise caption / caveat |
|---|---|---|
| Table: preliminary runtime problem | Phase 1 B2/B3/QSAL s/batch and reported epoch estimates | “Historical projection with unresolved batch-size denominator”; **not measured epoch** |
| Figure: repeated quantum QNode dispatch | **16→1 B3**, **384→3 QSAL** on T=2 | “Python-level QNode invocation counts, not logical circuit/statevector operations” |
| Table: isolated batching effect | Jobs **7934/7937** paired forward medians | Same legacy backend, synthetic T=2, only three forward timings |
| Table: numerical correctness and one-step latency | Jobs **8255/8256**, output/gradient/update differences and walltime | Real CASE, paired one optimizer step; no extrapolation to entire epoch |
| Table: steady short candidate timing | Jobs **8260/8261** (B=2) and **8262** (B=8) | Warm-up and repetition protocols shown explicitly |
| Figure: final optimised epoch cost | Job **8306**, **451 s B3 / 2164 s QSAL** | Complete entrypoint, full A100, **optimised-only** |
| Table: quality and training status | Train/validation loss, checkpoints, model parameter counts | One epoch; not a statistically valid quality comparison |
| Table: future work | O0–O8 plan/status and §11 priorities | Distinguish implemented, explored, deferred and architecture-changing proposals |

### 20.1 Mandatory wording checks before citing speedups

1. **Always attach a job ID and timing scope** to a reported speedup.
2. **Always distinguish simulator configuration from QNode batching**; neither must be represented as a change to quantum model capacity.
3. **Never describe the 184.919/0.511-second QSAL pair as “362× faster full training.”** It is one unusually expensive reference step versus one candidate step.
4. **Never compare the Phase 1 15.02h / 31.81h projections with 8306 as a matched experiment.** The benchmark denominator and GPU/timing protocols differ.
5. **Never use job 7898 as a corrected eight-modality QSAL baseline.** It predates the source propagation fix.
6. **Do not report thirty-epoch learning curves or multi-seed statistical error bars** until those experiments have actually run.
7. **If a source file or proposed archive cannot be found on the HPC filesystem**, mark it as not yet archived rather than asserting it exists.

### 20.2 Compact recommended results sentence

> In a bounded T=2 controlled profile, replacing serial QNode dispatch with broadcasted PQC inputs reduced QNode calls from 16 to one (B3) and 384 to three (QSAL), improving forward medians by factors of 1.317 and 1.452 (jobs 7934 and 7937). Paired real-CASE one-optimizer-step comparisons between the legacy and adopted simulator/differentiation paths achieved 5.91× (B3, job 8255) and approximately 361.9× (QSAL, job 8256) step-level speed ratios while satisfying the numerical correctness gates; neither ratio is an epoch claim. Finally, job 8306 completed the actual optimised CASE training and validation epochs on one full NVIDIA A100 80GB GPU in 451 s for B3 and 2164 s for QSAL, with saved checkpoints and finite losses. The historical reference and current epoch results do not share a fully matched timing protocol and are therefore not used to claim a controlled end-to-end speedup.

