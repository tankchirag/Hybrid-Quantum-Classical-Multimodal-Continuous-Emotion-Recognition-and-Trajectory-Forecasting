# datasets/audit/dataset_auditor.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class DatasetAuditor(ABC):
    """
    Base interface for all dataset auditors.

    Every dataset auditor must implement:
        - dataset discovery
        - integrity validation
        - statistical auditing
        - report generation

    Examples:
        CASE
        OMG
        MuSe
        Custom datasets

    This class intentionally remains dataset agnostic.
    """

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)

    @abstractmethod
    def discover(self) -> Dict[str, Any]:
        """
        Discover dataset files and structure.

        Returns
        -------
        Dict[str, Any]
            Dataset discovery metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def audit(self) -> Dict[str, Any]:
        """
        Execute audit procedure.
        """
        raise NotImplementedError