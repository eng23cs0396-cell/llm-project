from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from time import perf_counter

from sqlalchemy import create_engine, text


@dataclass
class ExecutionResult:
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    execution_ms: float


class SQLExecutor:
    def execute(
        self,
        db_path: str | None,
        sql: str,
        max_rows: int = 1000,
        timeout_seconds: float = 5.0,
        db_type: str = "sqlite",
        connection_url: str | None = None,
    ) -> ExecutionResult:
        if connection_url:
            return self._execute_sqlalchemy(
                connection_url=connection_url,
                sql=sql,
                max_rows=max_rows,
            )

        if not db_path:
            raise ValueError("db_path is required for file-backed database execution")

        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
        try:
            conn.row_factory = sqlite3.Row
            start = perf_counter()
            cursor = conn.execute(sql)

            fetched = cursor.fetchmany(max_rows + 1)
            elapsed_ms = (perf_counter() - start) * 1000

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            truncated = len(fetched) > max_rows
            rows = fetched[:max_rows]
            serialized_rows = [list(row) for row in rows]

            return ExecutionResult(
                columns=columns,
                rows=serialized_rows,
                row_count=len(serialized_rows),
                truncated=truncated,
                execution_ms=elapsed_ms,
            )
        finally:
            conn.close()

    def _execute_sqlalchemy(
        self,
        connection_url: str,
        sql: str,
        max_rows: int,
    ) -> ExecutionResult:
        engine = create_engine(connection_url)
        try:
            start = perf_counter()
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                fetched = result.fetchmany(max_rows + 1)

            elapsed_ms = (perf_counter() - start) * 1000
            columns = list(result.keys())
            truncated = len(fetched) > max_rows
            rows = fetched[:max_rows]
            serialized_rows = [list(row) for row in rows]

            return ExecutionResult(
                columns=columns,
                rows=serialized_rows,
                row_count=len(serialized_rows),
                truncated=truncated,
                execution_ms=elapsed_ms,
            )
        finally:
            engine.dispose()
