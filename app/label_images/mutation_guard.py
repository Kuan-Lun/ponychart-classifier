"""Shared fail-closed guard for every rawimage mutation entry point."""

import dataclasses
from pathlib import Path

from .retirement_journal import ensure_no_pending_retirement


@dataclasses.dataclass(frozen=True)
class RawImageMutationGuard:
    """Block mutations while a crash-recovery journal owns rawimage state."""

    image_dir: Path

    def ensure_allowed(self) -> None:
        """Raise before mutation when retirement recovery is unresolved."""
        ensure_no_pending_retirement(self.image_dir)
