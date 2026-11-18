"""Reusable, import-safe runtime profiling infrastructure.

The package separates configuration, model adapters, execution, instrumentation,
validation, reporting, and command-line handling. Importing the package does not
load production models, construct PennyLane QNodes, or initialize CUDA.

The package provides measurement infrastructure only. It does not establish
realistic CASE training performance or validate T=120 execution.
"""

__all__: list[str] = []
