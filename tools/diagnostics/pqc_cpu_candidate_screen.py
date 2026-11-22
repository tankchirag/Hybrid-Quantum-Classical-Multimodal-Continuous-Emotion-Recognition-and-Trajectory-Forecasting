"""Small CPU backend correctness and timing screen for UnifiedPQC."""

import time

import pennylane as qml
import torch

from quantum.unified_pqc import UnifiedPQC


CANDIDATES = (
    ("reference", "lightning.qubit", "best"),
    ("adjoint", "lightning.qubit", "adjoint"),
    ("torch_backprop", "default.qubit", "backprop"),
)


def evaluate(qnode, angles, theta, phi):
    result = qnode(angles.reshape(-1, 4), theta, phi)
    outputs = torch.stack(result, dim=-1).reshape(*angles.shape)

    gradients = torch.autograd.grad(
        outputs.square().sum(),
        (angles, theta, phi),
    )

    return outputs, *gradients


def run():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    model = UnifiedPQC(
        num_qubits=4,
        depth=3,
        device_name="lightning.qubit",
    ).double()

    reference_theta = model.theta.detach().clone()
    reference_phi = model.phi.detach().clone()

    small_angles = torch.randn(
        2, 3, 4, dtype=torch.float64
    ) * 0.2

    reference = None

    for name, backend, method in CANDIDATES:
        print(f"\n=== {name}: {backend} / {method} ===", flush=True)

        try:
            device = qml.device(backend, wires=4)
            qnode = qml.QNode(
                model._circuit,
                device,
                interface="torch",
                diff_method=method,
            )

            def execute(source):
                angles = source.detach().clone().requires_grad_(True)
                theta = reference_theta.clone().requires_grad_(True)
                phi = reference_phi.clone().requires_grad_(True)
                return evaluate(qnode, angles, theta, phi)

            values = execute(small_angles)

            if not all(torch.isfinite(v).all() for v in values):
                raise RuntimeError("Non-finite outputs or gradients")

            print("Dtypes:", [str(v.dtype) for v in values])

            if reference is None:
                reference = values
                print("Correctness: REFERENCE")
            else:
                errors = [
                    (v.double() - r.double()).abs().max().item()
                    for v, r in zip(values, reference)
                ]

                print("Maximum errors:", errors)

                valid = all(
                    torch.allclose(
                        v.double(),
                        r.double(),
                        rtol=1e-5,
                        atol=1e-6,
                    )
                    for v, r in zip(values, reference)
                )

                print("Correctness:", "PASS" if valid else "FAIL")

                if not valid:
                    continue

            # Small timing screen; not a real training benchmark.
            timing_angles = torch.randn(
                2, 10, 4, dtype=torch.float64
            ) * 0.2

            execute(timing_angles)  # Warm-up

            durations = []

            for _ in range(3):
                start = time.perf_counter()
                execute(timing_angles)
                durations.append(time.perf_counter() - start)

            print("Step times (s):", durations)
            print("Median (s):", sorted(durations)[1])

        except Exception as exc:
            print(
                f"FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    run()
