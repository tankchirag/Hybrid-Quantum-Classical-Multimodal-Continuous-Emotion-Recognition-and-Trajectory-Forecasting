from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime

from trace_core import (
    TraceRecorder,
    entry_to_markdown,
    build_mermaid,
)


# ==========================================================
# Directory
# ==========================================================

def ensure_report_dir(
    path: str | Path,
) -> Path:

    report_dir = Path(path)

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return report_dir


# ==========================================================
# TXT REPORT
# ==========================================================

def write_txt_report(
    recorder: TraceRecorder,
    output_file: str | Path,
) -> None:

    output_file = Path(output_file)

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as fp:

        fp.write(
            "=" * 80 + "\n"
        )

        fp.write(
            f"{recorder.model_name} "
            "ARCHITECTURE AUDIT\n"
        )

        fp.write(
            "=" * 80 + "\n\n"
        )

        if recorder.metadata:

            fp.write(
                "CONFIGURATION\n"
            )

            fp.write(
                "-" * 80 + "\n"
            )

            for k, v in recorder.metadata.items():

                fp.write(
                    f"{k}: {v}\n"
                )

            fp.write("\n")

        for entry in recorder.entries:

            fp.write(
                "-" * 80 + "\n"
            )

            fp.write(
                f"STAGE    : {entry.stage}\n"
            )

            fp.write(
                f"MODULE   : {entry.module}\n"
            )

            fp.write(
                f"INPUT    : {entry.input_shape}\n"
            )

            fp.write(
                f"OUTPUT   : {entry.output_shape}\n"
            )

            if entry.theory:

                fp.write(
                    f"THEORY   : {entry.theory}\n"
                )

            if entry.purpose:

                fp.write(
                    f"PURPOSE  : {entry.purpose}\n"
                )

            if entry.notes:

                fp.write(
                    f"NOTES    : {entry.notes}\n"
                )

            fp.write(
                f"STATUS   : {entry.status}\n\n"
            )

        fp.write(
            "=" * 80 + "\n"
        )

        fp.write(
            f"PASS   : {recorder.passed_count()}\n"
        )

        fp.write(
            f"FAIL   : {recorder.failed_count()}\n"
        )

        fp.write(
            f"RESULT : "
            f"{'PASS' if recorder.success() else 'FAIL'}\n"
        )

        fp.write(
            "=" * 80 + "\n"
        )


# ==========================================================
# CSV REPORT
# ==========================================================

def write_csv_report(
    recorder: TraceRecorder,
    output_file: str | Path,
) -> None:

    output_file = Path(output_file)

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.writer(fp)

        writer.writerow(
            [
                "stage",
                "module",
                "input",
                "output",
                "theory",
                "purpose",
                "notes",
                "status",
            ]
        )

        for entry in recorder.entries:

            writer.writerow(
                [
                    entry.stage,
                    entry.module,
                    str(entry.input_shape),
                    str(entry.output_shape),
                    entry.theory,
                    entry.purpose,
                    entry.notes,
                    entry.status,
                ]
            )


# ==========================================================
# MARKDOWN REPORT
# ==========================================================

def write_markdown_report(
    recorder: TraceRecorder,
    output_file: str | Path,
) -> None:

    output_file = Path(output_file)

    md = []

    md.append(
        f"# {recorder.model_name} Architecture Audit"
    )

    md.append("")

    md.append(
        f"Generated: {datetime.now()}"
    )

    md.append("")

    if recorder.metadata:

        md.append(
            "## Configuration"
        )

        md.append("")

        for k, v in recorder.metadata.items():

            md.append(
                f"- **{k}**: {v}"
            )

        md.append("")

    md.append(
        "## Runtime Module Execution"
    )

    md.append("")

    for idx, module in enumerate(
        recorder.executed_modules,
        start=1,
    ):

        md.append(
            f"{idx}. {module}"
        )

    md.append("")

    md.append(
        "## Detailed Audit"
    )

    md.append("")

    for entry in recorder.entries:

        md.append(
            entry_to_markdown(
                entry
            )
        )

    md.append("")

    md.append(
        "## Summary"
    )

    md.append("")

    md.append(
        f"- PASS: {recorder.passed_count()}"
    )

    md.append(
        f"- FAIL: {recorder.failed_count()}"
    )

    md.append(
        f"- RESULT: "
        f"{'PASS' if recorder.success() else 'FAIL'}"
    )

    output_file.write_text(
        "\n".join(md),
        encoding="utf-8",
    )


# ==========================================================
# MERMAID REPORT
# ==========================================================

def write_mermaid_report(
    nodes,
    output_file: str | Path,
) -> None:

    output_file = Path(output_file)

    mermaid = build_mermaid(
        nodes
    )

    output_file.write_text(
        mermaid,
        encoding="utf-8",
    )


# ==========================================================
# THEORY VALIDATION REPORT
# ==========================================================

def write_theory_report(
    validations,
    output_file: str | Path,
) -> None:

    output_file = Path(output_file)

    lines = []

    lines.append(
        "# Theory vs Implementation"
    )

    lines.append("")

    for item in validations:

        lines.append(
            f"## {item['component']}"
        )

        lines.append("")

        lines.append(
            f"**Theory:** "
            f"{item['theory']}"
        )

        lines.append("")

        lines.append(
            f"**Runtime:** "
            f"{item['runtime']}"
        )

        lines.append("")

        lines.append(
            f"**Status:** "
            f"{item['status']}"
        )

        lines.append("")

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ==========================================================
# ALL REPORTS
# ==========================================================

def write_all_reports(
    recorder: TraceRecorder,
    report_dir: str | Path,
    mermaid_nodes=None,
    theory_results=None,
) -> None:

    report_dir = ensure_report_dir(
        report_dir
    )

    prefix = (
        recorder.model_name.lower()
        .replace(" ", "_")
    )

    write_txt_report(
        recorder,
        report_dir /
        f"{prefix}.txt",
    )

    write_csv_report(
        recorder,
        report_dir /
        f"{prefix}.csv",
    )

    write_markdown_report(
        recorder,
        report_dir /
        f"{prefix}.md",
    )

    if mermaid_nodes:

        write_mermaid_report(
            mermaid_nodes,
            report_dir /
            f"{prefix}_mermaid.md",
        )

    if theory_results:

        write_theory_report(
            theory_results,
            report_dir /
            f"{prefix}_theory.md",
        )

    print(
        f"\nReports saved to:\n"
        f"{report_dir.resolve()}"
    )