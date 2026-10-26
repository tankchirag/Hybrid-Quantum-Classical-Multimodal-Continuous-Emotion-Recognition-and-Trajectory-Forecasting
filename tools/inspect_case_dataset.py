# tools/inspect_case_dataset.py

from __future__ import annotations

import pathlib 
import sys
import os
from pathlib import Path
from networkx import config
from sympy import pprint

DATASET_ROOT = Path(
    r"D:\Datasets\CASE"
)

PROJECT_ROOT = (
    pathlib.Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from pathlib import Path

from datasets.audit.case_auditor import CaseAuditor

def main() -> None:

    auditor = CaseAuditor(
        dataset_root=DATASET_ROOT
    )

    report = auditor.audit()

    auditor.generate_reports(
        report
    )

    print(
        "\nCASE DATASET AUDIT\n"
    )

    pprint(report)

    print(
        "\nGenerated:\n"
        "reports/case_statistics.json\n"
        "docs/datasets/case_dataset_spec.md\n"
    )


if __name__ == "__main__":
    main()