"""ASE database wrapper with status tracking."""

import logging
from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.db import connect

logger = logging.getLogger(__name__)

STATUS_GENERATED = "generated"
STATUS_RUNNING = "running"
STATUS_CONVERGED = "converged"
STATUS_FAILED = "failed"


class DefectDB:
    """Wrapper around ASE SQLite database for defect workflows."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.db = connect(str(self.path))
        logger.info("Database: %s", self.path)

    def add_structure(self, atoms: Atoms, **kvp) -> int:
        kvp.setdefault("status", STATUS_GENERATED)
        row_id = self.db.write(atoms, key_value_pairs=kvp)
        return row_id

    def add_pristine(self, atoms: Atoms, label: str = "pristine") -> int:
        return self.add_structure(
            atoms, label=label, defect_type="pristine",
            distortion_type="none", distortion_mag=0.0,
        )

    def set_status(self, row_id: int, status: str):
        self.db.update(row_id, status=status)

    def store_relaxed(self, row_id: int, atoms: Atoms,
                      energy: float, n_steps: int,
                      final_fmax: float = 0.0, converged: bool = True):
        self.db.update(
            row_id, atoms=atoms, status=STATUS_CONVERGED,
            total_energy=energy, relax_steps=n_steps,
            final_fmax=final_fmax, is_converged=int(converged),
        )

    def mark_failed(self, row_id: int, reason: str = ""):
        self.db.update(row_id, status=STATUS_FAILED, fail_reason=reason)

    def get_by_status(self, status: str) -> list:
        rows = []
        for row in self.db.select(status=status):
            rows.append((row.id, row.toatoms(), row.key_value_pairs))
        return rows

    def get_all_defect(self) -> list:
        rows = []
        for row in self.db.select():
            if row.key_value_pairs.get("defect_type") != "pristine":
                rows.append((row.id, row.toatoms(), row.key_value_pairs))
        return rows

    def get_pristine(self) -> Optional[tuple]:
        for row in self.db.select(defect_type="pristine"):
            return (row.id, row.toatoms(), row.key_value_pairs)
        return None

    def get_converged(self) -> list:
        return self.get_by_status(STATUS_CONVERGED)

    def get_defect_configs(self) -> list[str]:
        """Return the list of distinct defect_config names in the DB.

        Preserves insertion order (first-seen wins). Pristine rows are
        excluded. Rows without a ``defect_config`` key (e.g. from older
        runs) are reported as ``'default'``.
        """
        seen: list[str] = []
        for row in self.db.select():
            kvp = row.key_value_pairs
            if kvp.get("defect_type") == "pristine":
                continue
            name = kvp.get("defect_config", "default")
            if name not in seen:
                seen.append(name)
        return seen

    def count(self, **sel) -> int:
        return self.db.count(**sel)

    def summary(self) -> dict:
        out = {}
        for s in (STATUS_GENERATED, STATUS_RUNNING, STATUS_CONVERGED, STATUS_FAILED):
            out[s] = self.count(status=s)
        out["pristine"] = self.count(defect_type="pristine")
        out["total"] = self.count()
        return out
