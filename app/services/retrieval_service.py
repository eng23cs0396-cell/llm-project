from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
import re

from app.services.embedding_service import EmbeddingService
from app.storage import DatabaseMetadata, FewShotExample


@dataclass
class RetrievalContext:
    relevant_tables: list[str]
    relevant_columns: dict[str, list[str]]
    few_shot_examples: list[FewShotExample]


class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def build_context(
        self,
        question: str,
        metadata: DatabaseMetadata,
        few_shot_examples: list[FewShotExample],
    ) -> RetrievalContext:
        table_names = list(metadata.schema["tables"].keys())
        scored_tables = self.embedding_service.score_similarity(question, table_names)
        relevant_tables = [name for name, score in scored_tables if score > 0.0][:5]

        # Lexical fallback: include tables that are explicitly referenced in the question,
        # even when embedding similarity is low.
        q_tokens = set(re.findall(r"[a-zA-Z_]+", question.lower()))
        for table in table_names:
            table_lower = table.lower()
            table_tokens = set(re.findall(r"[a-zA-Z_]+", table_lower))
            singular = table_lower[:-1] if table_lower.endswith("s") else table_lower
            if (
                table_lower in question.lower()
                or singular in q_tokens
                or len(q_tokens & table_tokens) > 0
            ) and table not in relevant_tables:
                relevant_tables.append(table)

        if not relevant_tables and table_names:
            relevant_tables = table_names[:2]

        relevant_tables = self._expand_related_tables(metadata, relevant_tables, max_tables=5)

        relevant_columns: dict[str, list[str]] = {}
        for table in relevant_tables:
            cols = [col["name"] for col in metadata.schema["tables"][table]["columns"]]
            scored_cols = self.embedding_service.score_similarity(question, cols)
            relevant_columns[table] = [name for name, score in scored_cols if score > 0.0][:8]
            if not relevant_columns[table]:
                relevant_columns[table] = cols[:5]

        few_shot = self._select_examples(question, few_shot_examples, k=3)
        return RetrievalContext(
            relevant_tables=relevant_tables,
            relevant_columns=relevant_columns,
            few_shot_examples=few_shot,
        )

    def _expand_related_tables(
        self,
        metadata: DatabaseMetadata,
        seed_tables: list[str],
        max_tables: int,
    ) -> list[str]:
        if not seed_tables:
            return seed_tables

        expanded: list[str] = list(seed_tables)
        table_data = metadata.schema.get("tables", {})

        def add_table(table_name: str) -> None:
            if table_name and table_name in table_data and table_name not in expanded:
                expanded.append(table_name)

        # Add parents referenced by each seed table.
        for table in list(expanded):
            for fk in table_data.get(table, {}).get("foreign_keys", []):
                add_table(fk.get("references_table", ""))

        # Add children that reference each seed table.
        for table in list(expanded):
            for candidate, info in table_data.items():
                for fk in info.get("foreign_keys", []):
                    if fk.get("references_table") == table:
                        add_table(candidate)

        return expanded[:max_tables]

    def nearest_table(self, name: str, metadata: DatabaseMetadata) -> str | None:
        tables = list(metadata.schema["tables"].keys())
        matches = get_close_matches(name, tables, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def nearest_column(self, table: str, name: str, metadata: DatabaseMetadata) -> str | None:
        table_info = metadata.schema["tables"].get(table)
        if not table_info:
            return None
        columns = [col["name"] for col in table_info["columns"]]
        matches = get_close_matches(name, columns, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def _select_examples(
        self, question: str, examples: list[FewShotExample], k: int
    ) -> list[FewShotExample]:
        candidate_texts = [example.question for example in examples]
        ranked = self.embedding_service.score_similarity(question, candidate_texts)
        selected_questions = [question for question, _ in ranked[:k]]
        return [example for example in examples if example.question in selected_questions]
