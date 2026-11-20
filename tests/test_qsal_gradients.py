import json
from pathlib import Path

import torch

from models.wrappers.qsal import QSAL
from training.base_training_script import to_namespace
from utils.config import load_config


def test_active_qsal_fusion_has_finite_multisource_gradients():
    torch.manual_seed(2026)
    root = Path(__file__).resolve().parents[1]
    config = to_namespace(
        load_config(str(root / "configs" / "case_qsal.yaml"))
    )
    model = QSAL(config)
    model.train()

    enabled_sources = [
        source
        for source, enabled in model.extractors.modalities.items()
        if enabled
    ]
    batch_size = 2
    time_steps = 2
    raw_inputs = {
        source: torch.randn(
            batch_size,
            time_steps,
            1,
            requires_grad=True,
        )
        for source in enabled_sources
    }
    embeddings = model.extractors(raw_inputs)

    retained_outputs = {}
    hook_handles = []
    for role in ("q_pqc", "k_pqc", "v_pqc"):
        module = getattr(model.fusion, role)

        def retain_output_gradient(_module, _inputs, output, role=role):
            output.retain_grad()
            retained_outputs[role] = output

        hook_handles.append(
            module.register_forward_hook(retain_output_gradient)
        )

    try:
        fused = model.fusion(embeddings)
        loss = fused.square().mean() + 0.1 * fused.mean()
        assert loss.ndim == 0
        loss.backward()
    finally:
        for handle in hook_handles:
            handle.remove()

    assert bool(torch.isfinite(fused).all())
    assert bool(torch.isfinite(loss))

    parameters = {
        "fusion.q_pqc.theta": model.fusion.q_pqc.theta,
        "fusion.q_pqc.phi": model.fusion.q_pqc.phi,
        "fusion.k_pqc.theta": model.fusion.k_pqc.theta,
        "fusion.k_pqc.phi": model.fusion.k_pqc.phi,
        "fusion.v_pqc.theta": model.fusion.v_pqc.theta,
        "fusion.v_pqc.phi": model.fusion.v_pqc.phi,
        "fusion.to_pqc_input.weight": model.fusion.to_pqc_input.weight,
        "fusion.to_pqc_input.bias": model.fusion.to_pqc_input.bias,
        "fusion.pqc_to_fused.weight": model.fusion.pqc_to_fused.weight,
        "fusion.pqc_to_fused.bias": model.fusion.pqc_to_fused.bias,
        "fusion.fused_to_model.weight": model.fusion.fused_to_model.weight,
        "fusion.fused_to_model.bias": model.fusion.fused_to_model.bias,
    }
    for source in enabled_sources:
        projection = model.extractors.encoders[source].projection
        parameters[f"extractors.encoders.{source}.projection.weight"] = (
            projection.weight
        )
        parameters[f"extractors.encoders.{source}.projection.bias"] = (
            projection.bias
        )

    parameter_rows = []
    for name, parameter in parameters.items():
        gradient = parameter.grad
        assert parameter.requires_grad, f"{name} must be trainable"
        assert gradient is not None, f"{name} has no gradient"
        all_finite = bool(torch.isfinite(gradient).all())
        norm = float(torch.linalg.vector_norm(gradient).cpu())
        max_abs = float(gradient.abs().max().cpu())
        norm_is_zero = norm == 0.0
        assert all_finite, f"{name} has a non-finite gradient"
        assert not norm_is_zero, f"{name} has an exactly zero gradient norm"
        if name.startswith("extractors.encoders."):
            source = name.split(".")[2]
            role = f"{source} feature-encoder projection"
        elif ".q_pqc." in name:
            role = "query-circuit trainable parameters"
        elif ".k_pqc." in name:
            role = "key-circuit trainable parameters"
        elif ".v_pqc." in name:
            role = "value-circuit trainable parameters"
        elif ".to_pqc_input." in name:
            role = "shared source-to-quantum input projection"
        elif ".pqc_to_fused." in name:
            role = "quantum-output-to-fused projection"
        else:
            role = "fused-to-model output projection"
        parameter_rows.append(
            {
                "name": name,
                "role": role,
                "shape": list(parameter.shape),
                "requires_grad": parameter.requires_grad,
                "grad_is_none": False,
                "all_finite": all_finite,
                "norm": norm,
                "max_abs": max_abs,
                "norm_is_zero": norm_is_zero,
            }
        )

    quantum_gradients = {
        role: output.grad.reshape(
            batch_size,
            time_steps,
            len(enabled_sources),
            model.fusion.pqc_output_dim,
        )
        for role, output in retained_outputs.items()
    }
    assert set(quantum_gradients) == {"q_pqc", "k_pqc", "v_pqc"}
    for role, gradient in quantum_gradients.items():
        assert bool(torch.isfinite(gradient).all()), (
            f"{role} output gradient is non-finite"
        )
        assert float(torch.linalg.vector_norm(gradient).cpu()) > 0.0, (
            f"{role} output gradient norm is exactly zero"
        )

    source_gradient_rows = []
    for source_index, source in enumerate(enabled_sources[:2]):
        source_row = {"source": source}
        for role in ("q_pqc", "k_pqc", "v_pqc"):
            gradient = quantum_gradients[role][:, :, source_index, :]
            all_finite = bool(torch.isfinite(gradient).all())
            assert all_finite, (
                f"{role} gradient for {source} is non-finite"
            )
            source_row[f"{role}_finite"] = all_finite
            source_row[f"{role}_norm"] = float(
                torch.linalg.vector_norm(gradient).cpu()
            )

        input_gradient = raw_inputs[source].grad
        assert input_gradient is not None, (
            f"raw input for {source} has no gradient"
        )
        assert bool(torch.isfinite(input_gradient).all()), (
            f"raw input gradient for {source} is non-finite"
        )
        source_row["input_gradient_finite"] = True
        source_row["input_gradient_norm"] = float(
            torch.linalg.vector_norm(input_gradient).cpu()
        )
        assert source_row["input_gradient_norm"] > 0.0, (
            f"raw input gradient norm for {source} is exactly zero"
        )
        source_gradient_rows.append(source_row)

    print(
        "QSAL_GRADIENT_REPORT="
        + json.dumps(
            {
                "loss": float(loss.cpu()),
                "loss_finite": bool(torch.isfinite(loss)),
                "fused_output_shape": list(fused.shape),
                "enabled_sources": enabled_sources,
                "parameter_gradients": parameter_rows,
                "multi_source_gradients": source_gradient_rows,
            },
            sort_keys=True,
        )
    )
