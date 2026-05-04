from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatabaseMetadata:
    db_id: str
    name: str
    source_type: str
    db_type: str
    file_path: Path | None
    connection_url: str | None
    schema: dict[str, Any]
    relationship_graph: dict[str, list[str]]
    created_at: float


@dataclass
class FewShotExample:
    question: str
    sql: str
    tags: set[str] = field(default_factory=set)


class InMemoryStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._databases: dict[str, DatabaseMetadata] = {}
        self._few_shot_examples: list[FewShotExample] = []

    def save_database(self, metadata: DatabaseMetadata) -> None:
        with self._lock:
            self._databases[metadata.db_id] = metadata

    def get_database(self, db_id: str) -> DatabaseMetadata | None:
        with self._lock:
            return self._databases.get(db_id)

    def list_examples(self) -> list[FewShotExample]:
        with self._lock:
            return list(self._few_shot_examples)


STORE = InMemoryStore()
