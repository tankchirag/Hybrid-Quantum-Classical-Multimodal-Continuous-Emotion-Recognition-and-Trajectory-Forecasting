# Runtime profiling package

This package separates profile execution from reference equivalence checks and
provides temporary, exception-safe instrumentation. It is infrastructure only:
**Task 3A does not validate T=120 runtime or establish realistic CASE training
acceleration.** Do not interpret the Task 2 `T=2` result as evidence of
full-length training speed-up.

## CLI

```text
python -m tools.runtime_profiling.cli --help
python -m tools.runtime_profiling.cli profile \
  --model b3 --config configs/case_b3.yaml --temporal-length 2 \
  --output artifacts/runtime_opt/b3-profile.json
python -m tools.runtime_profiling.cli compare \
  --model qsal --config configs/case_qsal.yaml --temporal-length 2 \
  --reference-json artifacts/runtime_opt/baseline.json \
  --output artifacts/runtime_opt/qsal-equivalence.json
```

`--temporal-length` is required and explicit; the package never silently takes
the configuration's observation length as a benchmark size. Profile defaults
are one warm-up, three measured forwards, seed 42, and one additional
forward/diagnostic-loss/backward pass. `--no-forward-backward` disables that
last pass. The separate `compare` command runs one forward/backward and accepts
`--atol` and `--rtol`; defaults are the historical Task 2 values `1e-5` and
`1e-4`. Outputs are not overwritten unless `--overwrite` is supplied.

The CLI parses arguments before importing execution modules or production model
wrappers. Importing the package or requesting `--help` does not instantiate
models, create QNodes, or query CUDA.

## Package architecture

- `config.py`: immutable run options, active YAML normalization, ordered
  enabled-source validation, and lazy model/synthetic-batch construction.
- `adapters.py`: centralized B3/QSAL wrapper, component, parameter, and circuit
  metadata mappings; production classes are imported only when instantiated.
- `instrumentation.py`: temporary module hooks, QNode call wrappers, timing
  records, and CUDA synchronization. Hooks/wrappers are restored on normal and
  exceptional context exit.
- `execution.py`: explicit warm-up/forward/backward phases. No optimizer is
  constructed or stepped.
- `equivalence.py`: numerical, manifest, and gradient comparisons plus a
  separate one-pass equivalence execution.
- `validation.py`: output/gradient snapshots, required-gradient checks,
  and source-input gradient checks.
- `reporting.py`: reproducibility metadata, historical JSON validation and
  adaptation, finite-only JSON serialization, and collision-safe output.
- `cli.py`: import-safe `profile` and `compare` subcommands.

## Timing and scientific semantics

Wall-clock durations use `time.perf_counter()` in seconds. CUDA is synchronized
immediately before and after each timed forward, component, QNode call, loss,
and backward interval. The first warm-up can be wrapped by
`torch.profiler` for the historical conversion-event counters; its wall time
therefore includes profiler overhead. Nested component timings are inclusive
and should not be summed. Forward repetitions run in evaluation mode under
`torch.no_grad()`; the optional single forward/backward uses training mode and
the Task 2 diagnostic scalar, not the configured training objective.

QNode invocation counts are Python calls to each existing QNode. They are
recorded separately from circuit-evaluation counts; circuit evaluations are
reported as unavailable unless a backend exposes a reliable counter. The
package does not infer evaluations from QNode calls or batch size.

Required gradients are sourced from the centralized model adapter and report
parameter name, trainability, missing status, finiteness, norm, maximum
absolute value, and exact-zero status. Numerical comparisons check names,
shapes, available dtypes, and values. Task 2 snapshots did not record per-tensor
dtypes; the historical adapter marks those comparisons as dtype-unverified
rather than inventing dtype evidence.

## JSON, cleanup, and limitations

Profile documents retain Task 2's `phase`, `execution`, `input`, `models`, and
per-model field names where applicable. New snapshots include dtype and
structured issue details; the `compare` command adds `phase`, `execution`, and
`equivalence` metadata around its result. Historical baseline, post, and
standalone equivalence files are schema-checked before adaptation. JSON output
is sorted, rejects NaN/Infinity, and uses exclusive creation unless overwrite
is explicit.

No datasets, loaders, optimizer updates, checkpoints, or training loops are
used. Lightweight tests may exercise tiny local PyTorch modules, but the
package's model-level behavioral equivalence is not established by those tests.
Realistic CASE acceleration and T=120 behavior require a separately approved
benchmark task.
