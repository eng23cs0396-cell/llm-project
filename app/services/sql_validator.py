from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from app.storage import DatabaseMetadata


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str]


class SQLValidator:
    def validate(self, sql: str, metadata: DatabaseMetadata) -> ValidationResult:
        issues: list[str] = []
        statement = self._parse(sql, issues, metadata)
        if statement is None:
            return ValidationResult(ok=False, issues=issues)

        self._check_read_only(statement, issues)
        self._check_single_statement(sql, issues)
        self._check_tables_and_columns(statement, metadata, issues)

        return ValidationResult(ok=len(issues) == 0, issues=issues)

    def _parse(self, sql: str, issues: list[str], metadata: DatabaseMetadata) -> exp.Expression | None:
        dialect_map = {
            "postgresql": "postgres",
            "mariadb": "mysql",
            "mssql": "tsql",
        }
        read_dialect = dialect_map.get(metadata.db_type, metadata.db_type or "sqlite")
        try:
            return sqlglot.parse_one(sql, read=read_dialect)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Syntax error: {exc}")
            return None

    def _check_read_only(self, statement: exp.Expression, issues: list[str]) -> None:
        if not isinstance(statement, (exp.Select, exp.With)):
            issues.append("Only SELECT/CTE statements are allowed")

    def _check_single_statement(self, sql: str, issues: list[str]) -> None:
        parts = [part.strip() for part in sql.split(";") if part.strip()]
        if len(parts) > 1:
            issues.append("Only one SQL statement is allowed")

    def _check_tables_and_columns(
        self,
        statement: exp.Expression,
        metadata: DatabaseMetadata,
        issues: list[str],
    ) -> None:
        available_tables = set(metadata.schema["tables"].keys())
        available_tables_lc = {table.lower(): table for table in available_tables}
        table_columns_lc = {
            table.lower(): {str(column["name"]).lower() for column in info["columns"]}
            for table, info in metadata.schema["tables"].items()
        }

        select_aliases_lc: set[str] = set()
        for select_expr in statement.find_all(exp.Select):
            for projection in select_expr.expressions:
                alias = projection.alias
                if alias:
                    select_aliases_lc.add(alias.lower())

        alias_to_table_lc: dict[str, str] = {}
        for table_expr in statement.find_all(exp.Table):
            real_name = table_expr.name
            alias_name = table_expr.alias
            if not real_name or not alias_name:
                continue
            real_name_lc = real_name.lower()
            if real_name_lc in available_tables_lc:
                alias_to_table_lc[alias_name.lower()] = real_name_lc

        referenced_tables = {
            table.name for table in statement.find_all(exp.Table) if table.name
        }
        for table in referenced_tables:
            if table.lower() not in available_tables_lc:
                issues.append(f"Unknown table: {table}")

        referenced_columns = [column for column in statement.find_all(exp.Column)]
        for column in referenced_columns:
            col_name = column.name
            table_name = column.table
            if not col_name:
                continue
            col_name_lc = col_name.lower()
            if table_name:
                table_lc = table_name.lower()
                table_lc = alias_to_table_lc.get(table_lc, table_lc)
                if col_name_lc in select_aliases_lc:
                    continue
                if table_lc in table_columns_lc and col_name_lc not in table_columns_lc[table_lc]:
                    issues.append(f"Unknown column: {table_name}.{col_name}")
            else:
                if col_name_lc in select_aliases_lc:
                    continue
                if not any(col_name_lc in cols for cols in table_columns_lc.values()):
                    issues.append(f"Unknown column: {col_name}")
