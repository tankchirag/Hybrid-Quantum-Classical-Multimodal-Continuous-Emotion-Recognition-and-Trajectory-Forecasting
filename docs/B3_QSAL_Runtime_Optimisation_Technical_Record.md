# B3/QSAL Runtime Optimisation: Technical Record and Reproducible HPC Runbook

**Project:** Hybrid Quantum-Classical Multimodal Continuous Emotion Recognition and Trajectory Forecasting  
**Workstream:** CASE B3/QSAL runtime optimisation  
**Working branch:** `perf/b3-qsal-runtime-opt`  
**Worktree:** `~/projects/Hybrid-Quantum-Classical-Multimodal-Continuous-Emotion-Recognition-and-Trajectory-Forecasting-b3-qsal-opt`  
**Current captured allocation:** SLURM job `7742`  
**Document status:** Living engineering record. Update after every benchmark or optimisation commit.

---

## 1. Purpose

This document records what has actually been established for the B3/QSAL optimisation workstream, separates verified facts from assumptions, explains why the two quantum architectures are slow, and defines a safe optimisation and experiment-recording procedure.

The scientific constraint is strict: B1, B2, B3, and QSAL must retain the same dataset pipeline, modality encoders, history subsystem, temporal backbone, recognition head, forecasting head, losses, splits, precision policy, and target definitions. Only the fusion implementation may differ. Runtime work must preserve B3's unified entangled PQC semantics and QSAL's Q/K/V PQCs plus Gaussian attention.

## 2. Source-of-truth hierarchy

Use the following precedence when documents disagree:

1. Repository state and committed code at the recorded Git commit.
2. Captured configuration, command, environment, and SLURM logs for the experiment.
3. Phase 1 completion report for completed CASE integration.
4. Architecture comparison and framework documents for shared design invariants.
5. Original Phase 1 specification for historical intent only.
6. Phase 2 OMG document for future work only.

The Phase 1 completion record states that CASE, B1, B2, B3, QSAL, history conditioning, forecasting, end-to-end training, and speed benchmarking passed. It also records the final CASE temporal setup as 2 Hz, observation/history length 120, and forecast horizon 20. The earlier Phase 1 design used shorter windows, so it is not authoritative for completed experiments.

## 3. Verified current workspace state

### 3.1 Git/worktree

The terminal evidence supplied for this record shows:

```text
* perf/b3-qsal-runtime-opt
+ phase_1_case_training_b1
+ phase_1_case_training_b2
  main
```

Interpretation:

- `perf/b3-qsal-runtime-opt` is checked out in the current worktree.
- `+` indicates the B1 and B2 branches are checked out in other linked worktrees.
- The optimisation work is therefore isolated from B1 and B2 at the working-tree level.

Before any new run, capture the exact state:

```bash
pwd
git status --short --branch
git branch -vv
git rev-parse HEAD
git rev-parse --show-toplevel
git worktree list
```

Do not claim that a benchmark is reproducible unless the commit hash, dirty state, config, command, environment, and allocated hardware are recorded.

### 3.2 Python environment

The active prompt shows environment name `hybrid-qemotion`. The name alone does not prove which interpreter or installation paths are active. Verify:

```bash
which python
python -c 'import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)'
python -m pip list --editable
python -m pip check
```

Any editable package path must point to the B3/QSAL optimisation worktree, not the B1 or B2 worktree.

### 3.3 SLURM allocation and copied logs

The supplied terminal output verifies:

```text
SLURM_JOB_ID=7742
```

The following files were copied successfully into an experiment-specific directory:

```text
benchmark_results/case_speed/manual_7742/b3_case_gpu.log
benchmark_results/case_speed/manual_7742/training_speed_gpu.log
```

This is a useful archival step, but it does not by itself prove that the quantum circuit executed on the GPU. A PyTorch model can be on CUDA while a PennyLane QNode or simulator still runs on CPU.

## 4. Scientific architecture invariants

All four models share the surrounding pipeline and produce a fused representation compatible with the same temporal model.

| Model | Fusion mechanism | Quantum component | Must remain unchanged during speed work |
|---|---|---|---|
| B1 | Classical self-attention | None | Shared input/output contract and temporal stack |
| B2 | Concatenation plus projection | None | Capacity-control role |
| B3 | Source compression, aggregation, unified entangled PQC, projection | Unified PQC | Encoding, reuploading, trainable rotations, entanglement, measurement, output semantics |
| QSAL | Quantum query, key and value generation plus Gaussian attention | Q-PQC, K-PQC, V-PQC | Q/K/V circuit semantics, attention equation, modality-token behavior |

Shared output contract:

```text
fusion output: (B, T, 256)
```

Quantum register width is independent from modality count. A 4-qubit experiment must remain a 4-qubit experiment after optimisation.

## 5. Historical CASE speed baseline

The completion report records a CUDA benchmark at batch size 2:

| Model | Seconds/batch | Reported hours/epoch | Relative to B2 |
|---|---:|---:|---:|
| B2 | 0.396 | 0.39 | 1.0x |
| B3 | 15.148 | 15.02 | about 38x |
| QSAL | 32.080 | 31.81 | about 81x |

These values are historical baselines, not proof of current performance. Reuse them only when the new run matches the relevant commit, configuration, batch size, temporal dimensions, precision, simulator, differentiation method, GPU model, and timing method.

## 6. Why B3 and QSAL are slow

### 6.1 GPU allocation does not guarantee GPU quantum simulation

There are several separate device decisions:

1. PyTorch tensors and classical modules may be on CUDA.
2. The PennyLane device may be CPU-backed.
3. The differentiation transform may create repeated circuit evaluations.
4. Python loops may launch one QNode execution for every batch/time/token item.
5. Tensor conversion between CPU and GPU can add synchronization and transfer overhead.

Therefore `nvidia-smi` activity and `model.to("cuda")` are necessary checks, but not sufficient evidence that the PQC is GPU accelerated.

### 6.2 Circuit-call multiplication

B3 uses one unified PQC after aggregation. QSAL uses separate Q, K, and V quantum operations over modality tokens. QSAL therefore naturally has more quantum evaluations than B3, before backward differentiation is considered.

The dominant runtime is likely a product of:

```text
PQC evaluations per batch
= batch elements
× time steps
× modality/token evaluations
× Q/K/V branches
× differentiation evaluations
× repeated layers or circuit calls
```

The exact count must be instrumented from the repository. Do not infer it from architecture diagrams alone.

### 6.3 Simulator and differentiation overhead

Statevector simulation cost grows with the quantum state dimension and circuit depth. Training further requires derivatives. Depending on the configured interface and differentiation method, backward computation can require substantially more work than one forward circuit evaluation.

### 6.4 Small kernels and Python launch overhead

Four-qubit circuits are mathematically small. If each sample is executed through Python in a loop, dispatch, framework conversion, autograd bookkeeping, and synchronization can dominate arithmetic. In that situation, a GPU can be underutilised even though a CUDA allocation is active.

### 6.5 Why multi-GPU is not the first solution

Multi-GPU data parallelism replicates the model and divides batches. It does not eliminate serial QNode loops, CPU simulation, inefficient differentiation, or host-device transfers. With batch size 2, splitting work across multiple GPUs can add more communication overhead than useful parallel work.

Use multiple GPUs only after one GPU is demonstrably saturated, the quantum path is batch-capable, a larger effective batch exists, correctness is verified, and scaling is measured. For the current workstream, single-GPU profiling and call-reduction are the priority.

## 7. Determine whether the quantum path is CPU or GPU backed

Run this inside the active allocation and save the result:

```bash
mkdir -p benchmark_results/case_speed/manual_${SLURM_JOB_ID}/environment

{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "job=${SLURM_JOB_ID:-none}"
  echo "pwd=$PWD"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "python=$(command -v python)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi
  python - <<'PY'
import torch
import pennylane as qml
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("pennylane", qml.__version__)
PY
} | tee benchmark_results/case_speed/manual_${SLURM_JOB_ID}/environment/runtime.txt
```

Then inspect the actual device construction and QNode definitions:

```bash
grep -RIn \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  -E 'qml\.device|QNode|@qml\.qnode|diff_method|interface=|lightning|default\.qubit|cuda' \
  models training configs tests scripts tools 2>/dev/null \
  | tee benchmark_results/case_speed/manual_${SLURM_JOB_ID}/quantum_device_audit.txt
```

Record, do not guess:

- PennyLane device name
- wire count
- shots or analytic mode
- interface
- differentiation method
- dtype/precision
- batch support
- number of QNode calls in B3 and QSAL
- tensor device before and after each QNode boundary

## 8. Profiling-first optimisation plan

### O0: Freeze the reference

Create a reference artifact from unchanged code and a deterministic fixed batch:

- input tensors
- targets
- model state dictionary
- output tensors
- total loss and loss components
- selected parameter gradients
- optimizer state before/after one step
- RNG seeds
- complete configuration

No optimisation is accepted without comparison against this reference.

### O1: Instrument phase timings

Measure separately:

```text
data loading
host-to-device transfer
encoders
fusion classical preprocessing
PQC forward
attention/aggregation
transformer and heads
loss
backward
optimizer step
```

Synchronise CUDA immediately before and after measured GPU regions. Use warm-up iterations before recording.

### O2: Count quantum evaluations

Wrap or instrument the quantum execution boundary. Report QNode executions per:

- forward pass
- backward pass
- optimizer step
- complete batch

This count is a first-class benchmark metric. A speed-up caused by fewer repeated but mathematically identical evaluations is desirable. A speed-up caused by deleting required circuit operations is an architecture change and must be rejected.

### O3: Remove avoidable transfers and reconstruction

Safe targets include:

- move constant tensors and observables out of the inner loop
- avoid repeated `.cpu()`, `.numpy()`, and `torch.tensor(existing_tensor)` calls
- avoid rebuilding devices/QNodes inside `forward`
- avoid repeated dtype conversions
- keep outputs on the expected autograd/device path

### O4: Batch or broadcast equivalent circuit evaluations

Vectorise over independent batch/time/token dimensions only when the resulting circuit and measurements are semantically identical. Validate outputs and gradients against O0.

### O5: Remove duplicated circuit calls

Cache only values that are mathematically reusable within the same forward pass and remain connected to autograd. Never cache across optimizer steps when trainable circuit parameters have changed.

### O6: Evaluate simulator/backend options

A backend change is allowed only as a runtime implementation change, not a model change. Validate numerical equivalence and gradient behavior. Record backend, version, precision, interface, and differentiation method in every result.

### O7: Mixed precision

Apply mixed precision first to classical components. Quantum simulation may require or internally use a different precision. Accept only if output, loss, gradients, and training stability stay within predefined tolerances.

### O8: Multi-GPU, only after single-GPU optimisation

Test one optimization at a time:

1. Single GPU, current implementation.
2. Single GPU, batched/optimized quantum path.
3. Multi-GPU data parallelism with a batch large enough to distribute.
4. Compare throughput, memory, communication overhead, and numerical equivalence.

## 9. Correctness gates

Each optimisation commit must pass all gates:

### Gate A: Architecture

- same model identifier
- same fusion semantics
- same 4-qubit configuration for the 4Q benchmark
- same circuit depth and entanglement
- same measurements
- same history and temporal settings

### Gate B: Forward equivalence

On the fixed batch, save:

```text
max absolute error
mean absolute error
relative error
allclose result with explicit atol and rtol
```

### Gate C: Loss equivalence

Compare total, recognition, forecasting, CCC, and RMSE components.

### Gate D: Gradient equivalence

Compare representative classical and quantum parameter gradients. Also verify finite/non-null gradients for all trainable quantum parameters.

### Gate E: Optimizer-step equivalence

Starting from identical model and optimizer states, run one update and compare resulting parameters.

### Gate F: Performance

Require lower synchronized median seconds/batch across repeated measured runs. Report spread, not only the best result.

## 10. Benchmark protocol

Use a fixed protocol for every O0/O1/... comparison:

```text
same Git base and explicit optimisation commit
same CASE manifest and sample IDs
same model and qubit count
same circuit depth, shots, backend, diff method
same observation/history/horizon settings
same batch size
same precision
the same GPU model and allocation
same dataloader workers and CPU allocation
same warm-up count
same measured batch count
same random seed
```

Recommended minimum:

```text
warm-up batches: 5
measured batches: 20
independent repeats: 3
reported statistic: median plus min/max or IQR
```

Do not mix O0 and O1 logs in the same directory.

## 11. Run-directory standard

Use immutable per-run directories:

```text
benchmark_results/case_speed/
└── <model>_<variant>_<jobid>_<commit-short>/
    ├── metadata/
    │   ├── git.txt
    │   ├── slurm.txt
    │   ├── runtime.txt
    │   └── pip-freeze.txt
    ├── config/
    │   └── config-used.yaml
    ├── commands/
    │   └── benchmark-command.txt
    ├── logs/
    │   ├── stdout.log
    │   └── stderr.log
    ├── correctness/
    │   ├── outputs.pt
    │   ├── losses.json
    │   └── gradients.pt
    ├── profiling/
    │   ├── timings.csv
    │   └── qnode-counts.json
    └── results/
        ├── benchmark.csv
        ├── summary.json
        └── sacct.txt
```

The existing `manual_7742` directory should be retained as a historical capture. Do not rename it in a way that breaks references.

## 12. Self-identifying SLURM benchmark template

```bash
#!/usr/bin/env bash
#SBATCH --job-name=case-b3-qsal-opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/hpc/home/chandrayan/thesis-runs/b3-qsal-runtime-opt/logs/%x-%j.out
#SBATCH --error=/hpc/home/chandrayan/thesis-runs/b3-qsal-runtime-opt/logs/%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="$HOME/projects/Hybrid-Quantum-Classical-Multimodal-Continuous-Emotion-Recognition-and-Trajectory-Forecasting-b3-qsal-opt"
cd "$PROJECT_ROOT"

# Replace with the verified activation path for this worktree.
source .venv/bin/activate

COMMIT="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short HEAD)"
MODEL="${MODEL:-b3}"
VARIANT="${VARIANT:-O0}"
RUN_DIR="$PROJECT_ROOT/benchmark_results/case_speed/${MODEL}_${VARIANT}_${SLURM_JOB_ID}_${SHORT}"
mkdir -p "$RUN_DIR"/{metadata,config,commands,logs,correctness,profiling,results}

{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "pwd=$PWD"
  echo "branch=$(git branch --show-current)"
  echo "commit=$COMMIT"
  echo "job=${SLURM_JOB_ID}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  git status --short
} | tee "$RUN_DIR/metadata/git.txt"

# Stop reproducibility runs when tracked modifications exist.
git diff --quiet
git diff --cached --quiet

python --version > "$RUN_DIR/metadata/python-version.txt"
python -m pip freeze > "$RUN_DIR/metadata/pip-freeze.txt"
nvidia-smi > "$RUN_DIR/metadata/nvidia-smi.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/metadata/slurm.txt"

# Replace this with the repository's verified benchmark entry point.
printf '%q ' python -u tests/test_training_speed.py > "$RUN_DIR/commands/benchmark-command.txt"
printf '\n' >> "$RUN_DIR/commands/benchmark-command.txt"

srun python -u tests/test_training_speed.py 2>&1 | tee "$RUN_DIR/logs/benchmark.log"

sacct -j "$SLURM_JOB_ID" \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,AllocTRES \
  > "$RUN_DIR/results/sacct.txt"
```

The partition, account, memory, time, activation path, and benchmark entry point must match the actual cluster and repository. Do not copy placeholders into production submissions.

## 13. Git policy

One optimisation per commit:

```bash
git status --short --branch
git add <specific-files>
git commit -m "perf(b3): batch equivalent PQC evaluations"
git push
```

Suggested sequence:

```text
bench(quantum): add synchronized phase timing
bench(quantum): count qnode executions
perf(b3): remove repeated device transfers
perf(b3): batch equivalent circuit evaluations
perf(qsal): batch query circuit evaluation
perf(qsal): batch key circuit evaluation
perf(qsal): batch value circuit evaluation
test(quantum): compare outputs loss and gradients
report(quantum): document runtime optimization results
```

Do not combine backend changes, vectorisation, mixed precision, and model refactoring in one commit.

Once a commit is referenced by SLURM outputs or thesis results, prefer merging later B2 changes instead of rebasing published experiment history.

## 14. Experiment report template

Create one report per optimisation:

```markdown
# <Model> <Variant> Runtime Experiment

## Identity
- Status:
- Branch:
- Commit:
- Worktree:
- Job ID:
- Date:

## Scientific invariants
- Dataset/split:
- Model:
- Qubits/depth/entanglement:
- Backend/interface/diff method/shots:
- Observation/history/horizon:
- Precision:

## Change under test
- Hypothesis:
- Files changed:
- Expected semantic effect: none

## Correctness
- Output max/mean error:
- Loss difference:
- Gradient difference:
- Optimizer-step difference:
- Decision: PASS/FAIL

## Performance
- Warm-up/measured/repeats:
- Forward ms:
- Backward ms:
- Optimizer ms:
- Total sec/batch:
- Samples/sec:
- Peak GPU memory:
- QNode calls/batch:

## Resource evidence
- GPU:
- CPU allocation:
- MaxRSS:
- GPU utilization observations:

## Decision
- Keep/Reject/Repeat:
- Reason:
- Next isolated experiment:
```

## 15. Immediate next actions after job 7742

1. Keep the copied `manual_7742` logs unchanged.
2. Capture Git, Python, PennyLane, PyTorch, SLURM, and GPU metadata for job 7742 while the allocation still exists.
3. Inspect both logs for device name, CUDA availability, timings, warnings, and failures.
4. Audit `qml.device`, QNode definitions, interface, and differentiation method.
5. Add phase timing and QNode-call counting without changing model behavior.
6. Produce O0 reference outputs, loss, gradients, and optimizer-step artifacts.
7. Profile one B3 batch and one QSAL batch.
8. Select the largest measured bottleneck.
9. Implement exactly one optimisation.
10. Run correctness gates before accepting the speed result.

## 16. Current conclusions

- The worktree and branch isolation are in place.
- The supplied terminal evidence confirms an active allocation with job ID 7742 and successful archival of two benchmark logs.
- Historical results show B3 and QSAL are dramatically slower than B2 under the recorded CASE benchmark.
- It is not yet proven from the supplied terminal excerpt whether the PQCs are GPU-backed or CPU-backed.
- Multi-GPU training is not the recommended first intervention.
- The correct next step is single-GPU profiling, QNode-call counting, device/backend verification, and semantics-preserving batching or call reduction.
- No optimisation is accepted if it alters the fusion architecture, circuit semantics, data, temporal settings, precision policy, or shared infrastructure.

---

## 17. Project references used for this record

- `phase1_case_completion_report.md`: authoritative completed CASE implementation and historical benchmark.
- `architecture_comparison.md`: shared architecture and fusion-only comparison principle.
- `thesis_roadmap_plan.md`: universal, configuration-driven framework and M != Q principle.
- `thesis_expose_detailed.pdf`: B3 and QSAL circuit intent and controlled comparison.
- User-supplied terminal transcript: current branch, allocation, and log-copy evidence.

## 18. QSAL CASE source propagation correction (2026-11-17)

- **Confirmed defect:** CASE enables eight modalities and the QSAL feature extractor emits all eight, but `QSAL` previously constructed `QSALFusion` with `sources=None`. The fusion layer's legacy defaults matched only `ecg`, silently producing one source token.
- **Root cause and correction:** The wrapper now passes the enabled modality names from the normalized configuration in their original order. `QSALFusion` rejects empty or duplicate source lists and raises when configured feature embeddings are missing instead of silently dropping them.
- **Regression tests:** Four focused source-propagation tests pass: CASE order and `(B,T,8,128)` stacking, arbitrary generic names/order, missing sources, and duplicate sources.
- **Bounded real-CASE validation:** SLURM job `7919` completed one non-training forward on a real train-split batch of 2. Raw inputs had eight `(2,120,1)` tensors; extracted features had eight `(2,120,128)` tensors; selected QSAL tokens were `(2,120,8,128)`; fusion output was `(2,120,256)`. Recognition outputs were `(2,)` per continuous target and forecasting outputs `(2,20)` per target. B3 projected the same eight source names in the same order. No optimizer or training step ran.
- **Parameter count:** 8,121,553 trainable parameters before and after the correction.
- **Interpretation of job 7898:** Job `7898` is the original source-selection-defective QSAL runtime reference. Its QSAL run used the pre-correction wiring, which selected only ECG from the eight CASE modalities. It is not a corrected eight-modality QSAL runtime measurement and must not be presented as one. Preserve its scheduler/accounting evidence unchanged.

## 19. TASK 1 configuration-driven QSAL gradient verification (2026-11-17)

- **Date evidence:** The repository host and SLURM controller reported `2026-11-17` in `Europe/Berlin` (`+0100`); the task request identified `2026-11-15`. This discrepancy is recorded as reported. No system clock, scheduler record, historical filename, or prior evidence was changed to reconcile it.
- **Adjudication:** The former `CASE_SOURCES` constant and fixed `(2,3,8,128)` shape assertion were in the CASE-specific `test_case_configured_sources_are_preserved_in_order` fixture, not generic production QSAL code. That test already loaded `configs/case_qsal.yaml` and constructed `QSAL`, but its expected list had been independently defined. The test now derives its expected order and count from the wrapper's normalized modality mapping.
- **Configuration path:** `training/train_case_qsal.py` loads `configs/case_qsal.yaml`; `run_training` recursively normalizes it through `to_namespace`. The effective source key is `modalities`. The B3 entrypoint loads `configs/case_b3.yaml` directly. `load_config` reads one YAML file with `yaml.safe_load`; there is no config inheritance/merge in this path.
- **Effective configured sources:** Both active CASE configurations enable, in order, `ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap`. The QSAL wrapper filters the normalized modality mapping by enabled flag and passes that ordered list explicitly to `QSALFusion`.
- **Non-eight regression coverage:** A configuration-path test enables `skt, bvp, gsr`, disables `ecg`, and includes disabled/unconfigured embeddings in the input dictionary. It verifies configured order, exclusion, missing-source errors, and that no eight-source count is required. Separate checks reject an empty enabled list and duplicate configured names.
- **Differentiability check:** The focused gradient test runs the configured feature extractors and QSAL fusion in training mode on synthetic `(B=2,T=2)` inputs, forms a scalar loss from `(2,2,256)` fused output, and calls backward once. All required Q/K/V quantum parameters, the shared QSAL input projection, all enabled feature-encoder projections, and both post-quantum output projections had finite, non-`None`, nonzero-norm gradients. Retained Q/K/V outputs and source inputs showed finite gradients for both ECG and BVP. Loss was finite (`0.0851182267`). No optimizer step, checkpoint, training batch, or SLURM job was run.
- **Focused test results:** `tests/test_qsal_gradients.py`: 1 passed. `tests/test_qsal_source_propagation.py`: 6 passed. Only these focused tests were run; 13 dependency deprecation warnings were emitted per command.

## 20. TASK 2 batched QNode runtime profile (2026-11-17)

- **Date and scheduler evidence:** At task start, the repository host reported `2026-11-17T21:54:33+01:00` in `Europe/Berlin` (`CET`, UTC+01:00), hostname `zits-slurmlogin1.rz.fh-ingolstadt.de`. The SLURM controller was reachable on `zits-slurmlogin1` and reported the same date/time and offset. A final host check at `2026-11-17T22:12:39+01:00` confirmed the timezone. Successful jobs ran on `zits-slurmworker4.rz.fh-ingolstadt.de`: job `7934` from 22:02:51 to 22:03:42 and job `7937` from 22:07:24 to 22:08:30, both CET. No host/controller/compute-node date mismatch was observed. The earlier date note in §19 is historical and was not changed.
- **Run identity and resources:** Branch `perf/b3-qsal-runtime-opt`, base commit `2bd196bfde7f2a876c656518b0ddef5a30801080`. The `gpu` partition allocation used 4 CPUs, 16 GiB RAM, one GPU request, and a 25-minute limit. `nvidia-smi` identified an NVIDIA A100 80GB PCIe with a MIG instance (28 SM, about 20 GiB). The MIG utilization sample was `N/A`. CUDA runtime was 12.1; PyTorch was `2.1.0+cu121`.
- **Configuration and input:** The harness loaded `configs/case_b3.yaml` and `configs/case_qsal.yaml` through the active model/configuration paths. Both used batch 8, four qubits, depth 3, eight ordered configured sources (`ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap`), float32, model/input tensors on `cuda:0`, history length 120, and forecast horizon 20. Fixed synthetic inputs used temporal length 2 to bound the experiment. The QNodes used PennyLane `lightning.qubit`, Torch interface, `diff_method="best"`, and analytic shots (`Shots(total=None)`).
- **Protocol and commands:** Jobs `7934` (baseline) and `7937` (post) each performed one profiled warm-up forward, exactly three measured forwards, and one forward-plus-backward per model. The post job additionally ran one forward/backward equivalence check per model. No optimizer step, epoch, dataset preprocessing, or checkpoint was run. Commands:
  ```text
  sbatch artifacts/runtime_opt/task2_20261117_2154/profile_runtime.sbatch baseline
  sbatch artifacts/runtime_opt/task2_20261117_2154/profile_runtime.sbatch post
  ```
- **Measured forward wall time:** Individual measured passes and three-pass medians, in milliseconds:

  | Model | Baseline passes | Baseline median | Post passes | Post median | Median saved | Speed-up |
  |---|---:|---:|---:|---:|---:|---:|
  | B3 | 89.054, 86.290, 93.435 | 89.054 | 69.413, 67.642, 67.640 | 67.642 | 21.413 ms | 1.317x |
  | QSAL | 1875.455, 1717.533, 1709.737 | 1717.533 | 1213.073, 1171.569, 1183.204 | 1183.204 | 534.330 ms | 1.452x |

- **Forward component medians:** B3 feature extraction was 5.070 ms before / 5.114 ms after; fusion PQC was 71.772 / 51.487 ms; temporal backbone was 7.860 / 7.534 ms. QSAL feature extraction was 5.633 / 5.482 ms; classical `to_pqc_input` was 0.132 / 0.139 ms; Q, K, and V PQCs were 569.863 / 386.222 ms, 565.375 / 397.729 ms, and 565.147 / 383.797 ms respectively; Gaussian attention was 0.487 / 0.469 ms; the two following projections were 0.146 / 0.140 ms and 0.116 / 0.117 ms; temporal backbone was 7.892 / 7.750 ms. These are component medians from the same three measured forwards.
- **Forward-plus-backward samples:** B3 total was 1.323808 s before and 0.848950 s after (0.474858 s saved, 1.559x); QSAL total was 20.589187 s before and 19.653934 s after (0.935253 s saved, 1.048x). These are single samples, not stable medians. Backward alone was 1.171361 / 0.772759 s for B3 and 18.776151 / 17.851221 s for QSAL. The scalar diagnostic losses were finite: 2.958606 for B3 and 2.709494 for QSAL. They are profiling losses, not the configured training objective.
- **Bottleneck and optimization:** The original `UnifiedPQC.forward` iterated over each flattened batch/time sample and called the already-constructed QNode serially. It made 16 QNode calls for B3 and 128 calls in each QSAL Q/K/V branch (384 total). The implementation now sends the flattened batched angle tensor through the existing QNode once per module: one B3 call and one call for each of QSAL Q, K, and V. Device/QNode object counts remain one for B3 and three for QSAL; repeated construction was not the bottleneck. Circuit topology, trainable parameters, measurements, source order, outputs, Gaussian attention, loss, and training path were not changed. The relevant code is in `quantum/unified_pqc.py`.
- **Equivalence and gradients:** With fixed seeds and identical synthetic inputs, B3 and QSAL output shapes and trainable parameter manifests were identical before and after. Maximum output absolute difference was 0 for each model; maximum required-gradient absolute difference was `3.7253e-9` for B3 and `5.5879e-9` for QSAL, within `atol=1e-5`, `rtol=1e-4`. All outputs were finite. All 22 recorded B3 and 28 recorded QSAL required gradients were non-None, finite, and had nonzero full-tensor norms. ECG and BVP source-input gradients were finite and nonzero for both models. Parameter counts remained 8,073,056 (B3) and 8,121,553 (QSAL). No optimizer update was performed.
- **Memory and conversion observations:** Baseline and post peak CUDA memory matched exactly: B3 peak allocated/reserved `146,833,408 / 159,383,552` bytes; QSAL `147,060,736 / 159,383,552` bytes. The profiler did not record CPU resident memory. Synthetic inputs were created on the model device, so there was no dataloader-to-device interval; conversions were observed only via profiler events. Conversion event counts did not decrease materially (QSAL `aten::copy_` remained 16,929 calls; CUDA copy time was 17.843 ms before and 17.945 ms after), so the verified speed-up is attributed to reduced QNode dispatch, not fewer tensor copies. Quantum backward time was not isolated from full-model autograd. Circuit-evaluation counts were not independently available; only Python-level QNode invocation counts are reported.
- **Harness attempts and limitations:** Job `7930` failed before a forward because the harness queried an unavailable QNode `shots` attribute. Job `7932` completed B3 warm-up and three measured forwards, then backward failed because the history GRU was in eval mode; no QSAL profile ran. Job `7935` completed equivalence checks and B3 post profiling, then failed on the stale expected QNode call count (expected 16, observed 1); no QSAL post profile ran. These are retained as failed attempts, not successful measurements. The successful results use the same GPU node and harness, but the synthetic temporal length is 2 rather than the configured observation length 120; timing therefore establishes a bounded relative microbenchmark only and is not extrapolated to full-length sequences or epochs.
- **Artifacts:** The harness, submission script, baseline/post/equivalence JSON, and stdout/stderr for jobs `7930`, `7932`, `7934`, `7935`, and `7937` are in `artifacts/runtime_opt/task2_20261117_2154/`. The baseline JSON timestamp was `2026-11-17T22:03:40.669296+01:00`, the equivalence JSON file timestamp was 22:07:47 CET, and post JSON timestamp was `2026-11-17T22:08:28.415201+01:00`. Scheduler accounting reported `7934` and `7937` as `COMPLETED` with exit code `0:0`.
