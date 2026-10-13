from __future__ import annotations

from pathlib import Path


# ==========================================================
# Runtime Mermaid Generator
# ==========================================================

class RuntimeMermaidGenerator:
    """
    Converts runtime stage traces into Mermaid flowcharts.

    Example

        Raw Modalities
              ↓
        Encoders
              ↓
        QSAL Fusion
              ↓
        Transformer
              ↓
        Heads
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:

        self.model_name = model_name
        self.nodes = []

    # ------------------------------------------------------
    # Add Stage
    # ------------------------------------------------------

    def add_stage(
        self,
        stage_name: str,
        runtime_shape: str | None = None,
        description: str | None = None,
    ) -> None:

        self.nodes.append(
            {
                "stage": stage_name,
                "shape": runtime_shape,
                "description": description,
            }
        )

    # ------------------------------------------------------
    # Build Mermaid
    # ------------------------------------------------------

    def build_mermaid(
        self,
    ) -> str:

        lines = [
            "flowchart TD",
            ""
        ]

        for idx, node in enumerate(self.nodes):

            node_id = f"N{idx}"

            label = node["stage"]

            if node["shape"]:
                label += (
                    "<br/>"
                    f"{node['shape']}"
                )

            lines.append(
                f'{node_id}["{label}"]'
            )

        lines.append("")

        for idx in range(len(self.nodes) - 1):

            lines.append(
                f"N{idx} --> N{idx + 1}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------
    # Build Markdown
    # ------------------------------------------------------

    def build_markdown(
        self,
    ) -> str:

        return "\n".join(
            [
                f"# {self.model_name} Runtime Architecture",
                "",
                "Generated automatically from runtime execution.",
                "",
                "```mermaid",
                self.build_mermaid(),
                "```",
            ]
        )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    def save(
        self,
        output_file,
    ) -> Path:

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.build_markdown(),
            encoding="utf-8",
        )

        return output_path


# ==========================================================
# QSAL Diagram
# ==========================================================

def generate_qsal_runtime_mermaid(
    output_file,
):

    generator = RuntimeMermaidGenerator(
        model_name="QSAL"
    )

    generator.add_stage(
        "Raw Modalities"
    )

    generator.add_stage(
        "Feature Extractors",
        "(B,T,128)",
    )

    generator.add_stage(
        "Source Stack",
        "(B,T,N,128)",
    )

    generator.add_stage(
        "Quantum Projection",
        "(B,T,N,Q)",
    )

    generator.add_stage(
        "Reshape For PQC",
        "(B*T*N,1,Q)",
    )

    generator.add_stage(
        "Q-PQC",
        "(B*T*N,Q)",
    )

    generator.add_stage(
        "K-PQC",
        "(B*T*N,Q)",
    )

    generator.add_stage(
        "V-PQC",
        "(B*T*N,Q)",
    )

    generator.add_stage(
        "Restore Source Structure",
        "(B*T,N,Q)",
    )

    generator.add_stage(
        "Gaussian Attention",
        "(B*T,N,Q)",
    )

    generator.add_stage(
        "Source Pooling",
        "(B*T,Q)",
    )

    generator.add_stage(
        "Q → Fused",
        "(B*T,fused_dim)",
    )

    generator.add_stage(
        "Fused → Model",
        "(B*T,256)",
    )

    generator.add_stage(
        "Restore Temporal Shape",
        "(B,T,256)",
    )

    generator.add_stage(
        "Transformer Encoder",
        "(B,T,256)",
    )

    generator.add_stage(
        "Transformer Decoder",
        "(B,H,256)",
    )

    generator.add_stage(
        "Recognition Head",
    )

    generator.add_stage(
        "Forecasting Head",
    )

    return generator.save(
        output_file
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    output_file = (
        Path("trace_reports")
        / "qsal"
        / "qsal_runtime_mermaid.md"
    )

    saved_file = (
        generate_qsal_runtime_mermaid(
            output_file
        )
    )

    print(
        f"Saved: {saved_file}"
    )