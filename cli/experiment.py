"""Template Method base class for experiment CLIs.

All experiment CLIs share the same ``--run KEY`` / ``--report`` /
``--results-dir`` pattern.  Subclasses implement the hook methods that
define experiment-specific behaviour while this base owns the dispatch
logic and shared argparse setup.
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ponychart_classifier.training import configure_logging

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
"""Project-level directory for all experiment result JSONs."""


class ExperimentCLI(ABC):
    """Abstract base for ``--run`` / ``--report`` experiment CLIs."""

    def __init__(self) -> None:
        configure_logging(logging.INFO)
        self.logger = logging.getLogger(self.__class__.__module__)

    # ------------------------------------------------------------------
    # Hooks — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def experiment_name(self) -> str:
        """Short human-readable name (used in ``--help``)."""

    @abstractmethod
    def default_results_dir(self) -> Path:
        """Default directory for result JSONs."""

    @abstractmethod
    def available_keys(self) -> list[str]:
        """Valid keys accepted by ``--run``."""

    @abstractmethod
    def run_one(self, key: str, results_dir: Path) -> None:
        """Execute one experiment identified by *key* and save the result."""

    @abstractmethod
    def format_report(self, results_dir: Path) -> None:
        """Load all result JSONs from *results_dir* and print a report."""

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        """Override to register additional CLI flags."""

    # ------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=self.experiment_name())
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument(
            "--run",
            metavar="KEY",
            help=(
                "Run one experiment and save its result JSON. "
                f"Available: {', '.join(self.available_keys())}"
            ),
        )
        mode.add_argument(
            "--report",
            action="store_true",
            help="Print a comparison from all result JSONs in --results-dir.",
        )
        parser.add_argument(
            "--results-dir",
            type=Path,
            default=self.default_results_dir(),
            help=f"Where result JSONs live (default: {self.default_results_dir()})",
        )
        self.add_extra_args(parser)
        return parser.parse_args(argv)

    def main(self, argv: list[str] | None = None) -> None:
        self._args = self.parse_args(argv)
        if self._args.run is not None:
            self.run_one(self._args.run, self._args.results_dir)
        else:
            self.format_report(self._args.results_dir)
