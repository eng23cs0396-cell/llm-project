from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import networkx as nx
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL, make_url


class SchemaService:
    SUPPORTED_DIALECTS = {
        "sqlite",
        "postgresql",
        "mysql",
        "mariadb",
        "mssql",
        "oracle",
        "duckdb",
    }

    def analyze_sqlite_database(self, db_file: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        try:
            tables = self._list_tables(conn)
            schema: dict[str, Any] = {"tables": {}}
            graph = nx.DiGraph()

            for table in tables:
                columns = self._list_columns(conn, table)
                sample_rows = self._sample_rows(conn, table, limit=5)
                foreign_keys = self._list_foreign_keys(conn, table)

                schema["tables"][table] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys,
                    "sample_rows": sample_rows,
                }
                graph.add_node(table)

            for table, info in schema["tables"].items():
                for fk in info["foreign_keys"]:
                    parent = fk["references_table"]
                    if parent in schema["tables"]:
                        graph.add_edge(table, parent)

            relationship_graph = {
                node: list(graph.successors(node)) for node in graph.nodes
            }
            return schema, relationship_graph
        finally:
            conn.close()

    def analyze_database_url(self, db_url: str) -> tuple[dict[str, Any], dict[str, list[str]], str]:
        parsed: URL = make_url(db_url)
        dialect = parsed.get_backend_name()
        if dialect not in self.SUPPORTED_DIALECTS:
            supported = ", ".join(sorted(self.SUPPORTED_DIALECTS))
            raise ValueError(f"Unsupported database dialect '{dialect}'. Supported: {supported}")

        engine = create_engine(db_url)
        try:
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            schema: dict[str, Any] = {"tables": {}}
            graph = nx.DiGraph()

            for table in tables:
                cols = inspector.get_columns(table)
                pks = set(inspector.get_pk_constraint(table).get("constrained_columns") or [])
                columns = [
                    {
                        "name": col.get("name"),
                        "type": str(col.get("type", "")),
                        "not_null": not bool(col.get("nullable", True)),
                        "default": col.get("default"),
                        "primary_key": col.get("name") in pks,
                    }
                    for col in cols
                ]

                foreign_keys = [
                    {
                        "column": (fk.get("constrained_columns") or [""])[0],
                        "references_table": fk.get("referred_table", ""),
                        "references_column": (fk.get("referred_columns") or [""])[0],
                    }
                    for fk in inspector.get_foreign_keys(table)
                ]

                sample_rows = self._sample_rows_sqlalchemy(engine, table, limit=5)

                schema["tables"][table] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys,
                    "sample_rows": sample_rows,
                }
                graph.add_node(table)

            for table, info in schema["tables"].items():
                for fk in info["foreign_keys"]:
                    parent = fk["references_table"]
                    if parent in schema["tables"]:
                        graph.add_edge(table, parent)

            relationship_graph = {node: list(graph.successors(node)) for node in graph.nodes}
            return schema, relationship_graph, dialect
        finally:
            engine.dispose()

    def _list_tables(self, conn: sqlite3.Connection) -> list[str]:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]

    def _list_columns(self, conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return [
            {
                "name": row[1],
                "type": row[2],
                "not_null": bool(row[3]),
                "default": row[4],
                "primary_key": bool(row[5]),
            }
            for row in cursor.fetchall()
        ]

    def _list_foreign_keys(self, conn: sqlite3.Connection, table: str) -> list[dict[str, str]]:
        cursor = conn.execute(f"PRAGMA foreign_key_list({table})")
        return [
            {
                "column": row[3],
                "references_table": row[2],
                "references_column": row[4],
            }
            for row in cursor.fetchall()
        ]

    def _sample_rows(self, conn: sqlite3.Connection, table: str, limit: int) -> list[dict[str, Any]]:
        cursor = conn.execute(f"SELECT * FROM {table} LIMIT {limit}")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def _sample_rows_sqlalchemy(self, engine, table: str, limit: int) -> list[dict[str, Any]]:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table} LIMIT {limit}"))
            return [dict(row._mapping) for row in result.fetchall()]
