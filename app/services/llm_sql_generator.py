from __future__ import annotations

import json
import logging
import os
import re
from difflib import get_close_matches
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
import torch

logger = logging.getLogger(__name__)

from app.services.flan_t5_sql_config import (
    build_prompt as build_flan_t5_model_prompt,
    generation_kwargs as flan_t5_model_generation_kwargs,
)
from app.services.adapter_setup import setup_adapters
from app.services.mistral_7b_sql_config import (
    build_prompt as build_mistral_model_prompt,
    generation_kwargs as mistral_model_generation_kwargs,
)
from app.services.retrieval_service import RetrievalContext
from app.storage import DatabaseMetadata


@dataclass
class _ModelBundle:
    tokenizer: object
    model: object
    family: str
    model_path: str
    model_key: str


class LLMSQLGenerator:
    def __init__(self, model_path: str | list[str], fallback_generator, allow_fallback: bool = True) -> None:
        if isinstance(model_path, str):
            model_paths = [part.strip() for part in model_path.split(",") if part.strip()]
        else:
            model_paths = [str(path).strip() for path in model_path if str(path).strip()]
        self.model_paths = model_paths
        self.fallback_generator = fallback_generator
        self.allow_fallback = allow_fallback
        self._bundles: list[_ModelBundle] | None = None

    def warmup(self) -> None:
        self._load_bundles()

    # -----------------------------
    # 🚀 FAST PATH (CRITICAL)
    # -----------------------------
    def _handle_simple_query(self, question: str, metadata, context):
        """
        Heuristic: return True if the question is a simple projection/lookup
        that does not require joins, aggregation, ranking, or grouping.
        This no longer returns SQL directly; it signals the caller to prefer
        a Flan-only LLM path for simple queries.
        """
        q = question.lower()

        # If question mentions aggregates, joins, ranking — not simple
        if any(k in q for k in [
            "highest", "average", "avg", "group",
            "join", "top", "rank"
        ]):
            return False

        # Simple projection/listing intents
        if any(k in q for k in ["list all", "show all", "display all"]):
            return True

        if re.search(r"(names|name) of (all )?(\w+)", q) and context.relevant_tables:
            return True

        return False

    # -----------------------------
    # 🚨 GARBAGE FILTER
    # -----------------------------
    def _is_garbage_output(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in [
            "'agg'", "'conds'", "'sel'", "column_index",
            "operator_index", "value_index"
        ])

    def _clean_raw_output(self, text: str) -> str:
        """Strip dataset-specific junk tokens and normalize output before extraction."""
        # Remove common dataset markers
        text = re.sub(r"'(?:sel|agg|conds|where|group|having)'", "", text)
        text = re.sub(r"(?:column|operator|value)_index", "", text)
        text = re.sub(r"\{[^}]*\}", "", text)  # Remove curly brace structures
        text = re.sub(r"\[SELECT.*?\]", "", text, flags=re.IGNORECASE)  # Remove bracketed SQL
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -----------------------------
    # DOMAIN GROUNDING CHECK
    # -----------------------------
    def _is_question_relevant(self, question: str, metadata) -> bool:
        q = question.lower()

        # collect schema words
        table_names = [name.lower() for name in list(metadata.schema.get("tables", {}).keys())]
        column_names = []

        for t in table_names:
            cols = metadata.schema["tables"][t]["columns"]
            column_names.extend([c["name"].lower() for c in cols])

        schema_words = set(table_names + column_names)
        schema_tokens: set[str] = set()
        for word in schema_words:
            schema_tokens.update(re.findall(r"[a-zA-Z]+", word))

        # extract keywords from question
        tokens = set(re.findall(r"[a-zA-Z_]+", q))
        plain_tokens = set(re.findall(r"[a-zA-Z]+", q))
        normalized_tokens = set()
        for tok in plain_tokens:
            normalized_tokens.add(tok)
            if tok.endswith("s") and len(tok) > 3:
                normalized_tokens.add(tok[:-1])
            else:
                normalized_tokens.add(f"{tok}s")

        # exact overlap check
        overlap = tokens & schema_words
        token_overlap = normalized_tokens & schema_tokens
        if len(overlap) > 0 or len(token_overlap) > 0:
            return True

        # fuzzy overlap fallback to reduce false negatives like "highest scorer"
        matches = []
        schema_word_list = list(schema_words)
        for token in tokens:
            matches += get_close_matches(token, schema_word_list, n=1, cutoff=0.8)

        schema_token_list = list(schema_tokens)
        for token in plain_tokens:
            matches += get_close_matches(token, schema_token_list, n=1, cutoff=0.85)

        return len(matches) > 0

    def _mentioned_tables_in_question(self, question: str, metadata: DatabaseMetadata) -> list[str]:
        q = question.lower()
        mentioned: list[str] = []
        for table in metadata.schema.get("tables", {}).keys():
            t = table.lower()
            singular = t[:-1] if t.endswith("s") else t
            if t in q or re.search(rf"\b{re.escape(singular)}\b", q):
                if table not in mentioned:
                    mentioned.append(table)
        return mentioned

    def _find_bridge_table(self, metadata: DatabaseMetadata, table_a: str, table_b: str) -> tuple[str, str, str] | None:
        table_map = metadata.schema.get("tables", {})
        for bridge in table_map.keys():
            if bridge in {table_a, table_b}:
                continue
            to_a = self._find_join_edge(metadata, bridge, table_a)
            to_b = self._find_join_edge(metadata, bridge, table_b)
            if to_a and to_b:
                fk_a = to_a[1] if to_a[0] == "l" else to_a[2]
                fk_b = to_b[1] if to_b[0] == "l" else to_b[2]
                if fk_a and fk_b and fk_a != fk_b:
                    return (bridge, fk_a, fk_b)
        return None

    def _find_table_id_column(self, metadata: DatabaseMetadata, table: str) -> str:
        table_info = metadata.schema.get("tables", {}).get(table, {})
        cols = [str(c.get("name", "")) for c in table_info.get("columns", [])]
        lowered = [c.lower() for c in cols]
        for original, lower in zip(cols, lowered):
            if lower == "id":
                return original
        candidate = f"{table.lower()}_id"
        for original, lower in zip(cols, lowered):
            if lower == candidate:
                return original
        for original, lower in zip(cols, lowered):
            if lower.endswith("_id"):
                return original
        return "id"

    def _pick_temporal_column(self, metadata: DatabaseMetadata, table: str) -> str:
        return self._pick_column_by_keywords(metadata, table, ["date", "time", "year", "semester", "term", "effective", "created", "updated"])

    def _pick_metric_column(self, metadata: DatabaseMetadata, table: str) -> str:
        return self._pick_column_by_keywords(metadata, table, ["amount", "hours", "salary", "score", "grade", "rating", "credits", "value", "cost", "price", "budget"]) 

    def _pick_country_like_column(self, metadata: DatabaseMetadata, table: str) -> str:
        return self._pick_column_by_keywords(metadata, table, ["country", "region", "state", "location"]) 

    def _schema_has_relation_for_keywords(self, metadata: DatabaseMetadata, keywords: list[str]) -> bool:
        tables = metadata.schema.get("tables", {})
        for table_name, table_info in tables.items():
            table_lower = str(table_name).lower()

            for fk in table_info.get("foreign_keys", []):
                fk_col = str(fk.get("column", "")).lower()
                ref_table = str(fk.get("references_table", "")).lower()
                if any(k in fk_col or k in ref_table or k in table_lower for k in keywords):
                    return True

            for col in table_info.get("columns", []):
                col_name = str(col.get("name", "")).lower()
                # Id-like columns usually represent relational links.
                if col_name.endswith("_id") and any(k in col_name for k in keywords):
                    return True

        return False

    def _schema_has_inline_keyword_column(self, metadata: DatabaseMetadata, keywords: list[str]) -> bool:
        tables = metadata.schema.get("tables", {})
        for table_info in tables.values():
            for col in table_info.get("columns", []):
                col_name = str(col.get("name", "")).lower()
                if col_name.endswith("_id"):
                    continue
                if any(k in col_name for k in keywords):
                    return True
        return False

    def _is_sql_acceptable(self, question: str, sql: str, metadata: DatabaseMetadata | None = None) -> bool:
        if not sql:
            return False

        q = question.lower()
        s = f" {sql.lower()} "

        def has_phrase(phrase: str) -> bool:
            if " " in phrase:
                return phrase in q
            return re.search(rf"\b{re.escape(phrase)}\b", q) is not None

        # Check if question asks for a threshold/comparison with a specific column
        has_threshold = bool(re.search(r"(older|younger|higher|lower|greater|less|more|fewer)\s+than", q))
        
        if has_threshold:
            # If question mentions "age" or "older/younger", the SQL MUST have WHERE clause with comparison
            asks_age = "age" in q or "older" in q or "younger" in q
            if asks_age and " where " not in s:
                return False
            
            # If asking for a specific numeric threshold, SQL must include WHERE with comparison
            has_explicit_numeric = bool(re.search(r">\s*\d+|<\s*\d+|=\s*\d+", q))
            if has_explicit_numeric and " where " not in s:
                return False

        has_explicit_literal = bool(re.search(r"'[^']+'|\b\d{4}\b", q))
        asks_filtered_subset = any(
            has_phrase(k)
            for k in ["after", "before", "where", "year", "semester", "active", "since", "from"]
        ) or has_explicit_literal

        # --- FILTER CHECK ---
        if asks_filtered_subset:
            if " where " not in s:
                return False

        # --- COUNT CHECK ---
        if any(has_phrase(k) for k in ["count", "how many", "total", "number of"]):
            if "count(" not in s:
                return False

        # --- AVG CHECK ---
        if any(has_phrase(k) for k in ["average", "avg", "mean"]):
            if "avg(" not in s:
                return False

        # --- JOIN CHECK ---
        join_terms = ["along with", "with their", "and their", "enrolled in"]
        has_join_phrase = any(has_phrase(k) for k in join_terms)

        advisor_requires_join = False
        if has_phrase("advisor"):
            if metadata is None:
                advisor_requires_join = True
            else:
                has_advisor_relation = self._schema_has_relation_for_keywords(
                    metadata,
                    ["advisor", "instructor", "teacher", "mentor", "faculty"],
                )
                has_inline_advisor = self._schema_has_inline_keyword_column(
                    metadata,
                    ["advisor", "mentor", "instructor", "teacher"],
                )
                advisor_requires_join = has_advisor_relation and not has_inline_advisor

        temporal_enrollment_phrase = (
            has_phrase("enrolled in") and any(has_phrase(k) for k in ["year", "semester", "term", "after", "before", "since", "from"])
        )
        if (has_join_phrase or advisor_requires_join) and not temporal_enrollment_phrase:
            if " join " not in s:
                return False

        # --- PROJECTION CHECK ---
        asks_specific_projection = bool(
            re.search(r"\bnames?\b", q)
            or re.search(r"\bcredits?\b", q)
        )
        if asks_specific_projection:
            if "select *" in s:
                return False

        # --- ADVISOR INTENT CHECK ---
        if has_phrase("advisor") and " join " not in s:
            return False

        # --- GRADE INTENT CHECK ---
        select_match = re.search(r"(?is)\bselect\b\s+(.*?)\s+\bfrom\b", sql)
        select_text = select_match.group(1).lower() if select_match else ""

        grade_detail_request = (
            has_phrase("grade")
            and any(has_phrase(k) for k in ["along with", "with", "show", "list"])
            and not any(has_phrase(k) for k in ["average", "avg", "mean", "maximum", "max", "highest", "top", "second", "count"])
        )
        if grade_detail_request:
            if not any(token in select_text for token in ["grade", "score", "rating"]):
                return False

        # --- MAX INTENT CHECK ---
        asks_max = any(has_phrase(k) for k in ["maximum", "max", "highest", "top"])
        plain_max_grade = (
            asks_max
            and has_phrase("grade")
            and not any(has_phrase(k) for k in ["average", "avg", "mean", "second", "each", "per", "rank", "top 5"])
        )
        if plain_max_grade and "max(" not in s:
            return False

        return True

    def _extract_table_from_sql(self, sql: str) -> str:
        match = re.search(r"(?i)\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql)
        return match.group(1) if match else ""

    def _pick_primary_table(self, metadata: DatabaseMetadata, context: RetrievalContext) -> str:
        if context.relevant_tables:
            return context.relevant_tables[0]
        tables = list(metadata.schema.get("tables", {}).keys())
        return tables[0] if tables else ""

    def _pick_table_from_question(self, question: str, metadata: DatabaseMetadata, context: RetrievalContext) -> str:
        q = question.lower()
        tables = list(metadata.schema.get("tables", {}).keys())

        # Prefer explicit table mentions in the question.
        for table in tables:
            t = table.lower()
            singular = t[:-1] if t.endswith("s") else t
            if t in q or re.search(rf"\b{re.escape(singular)}\b", q):
                return table

        # Then fuzzy match question tokens to table names.
        tokens = set(re.findall(r"[a-zA-Z_]+", q))
        table_lowers = [t.lower() for t in tables]
        for token in tokens:
            matches = get_close_matches(token, table_lowers, n=1, cutoff=0.85)
            if matches:
                matched = matches[0]
                for original in tables:
                    if original.lower() == matched:
                        return original

        return self._pick_primary_table(metadata, context)

    def _pick_column_by_keywords(self, metadata: DatabaseMetadata, table: str, keywords: list[str]) -> str:
        table_info = metadata.schema.get("tables", {}).get(table, {})
        for col in table_info.get("columns", []):
            col_name = str(col.get("name", ""))
            col_lower = col_name.lower()
            if any(k in col_lower for k in keywords):
                return col_name
        return ""

    def _pick_name_like_column(self, metadata: DatabaseMetadata, table: str) -> str:
        name_col = self._pick_column_by_keywords(metadata, table, ["name", "title", "label"])
        if name_col:
            return name_col
        table_info = metadata.schema.get("tables", {}).get(table, {})
        for col in table_info.get("columns", []):
            col_name = str(col.get("name", ""))
            if col_name.lower() == "id" or col_name.lower().endswith("_id"):
                continue
            return col_name
        return ""

    def _find_table_with_metric_keyword(self, metadata: DatabaseMetadata, keywords: list[str]) -> tuple[str, str] | None:
        for table, info in metadata.schema.get("tables", {}).items():
            for col in info.get("columns", []):
                col_name = str(col.get("name", ""))
                lower = col_name.lower()
                if any(k in lower for k in keywords):
                    return table, col_name
        return None

    def _find_join_edge(
        self,
        metadata: DatabaseMetadata,
        left_table: str,
        right_table: str,
    ) -> tuple[str, str, str] | None:
        tables = metadata.schema.get("tables", {})
        left_id = self._pick_column_by_keywords(metadata, left_table, ["id"]) or "id"
        right_id = self._pick_column_by_keywords(metadata, right_table, ["id"]) or "id"

        for fk in tables.get(left_table, {}).get("foreign_keys", []):
            if str(fk.get("references_table", "")) == right_table:
                fk_col = str(fk.get("column", ""))
                ref_col = str(fk.get("references_column", "")) or right_id
                if fk_col and ref_col:
                    return ("l", fk_col, ref_col)

        for fk in tables.get(right_table, {}).get("foreign_keys", []):
            if str(fk.get("references_table", "")) == left_table:
                fk_col = str(fk.get("column", ""))
                ref_col = str(fk.get("references_column", "")) or left_id
                if fk_col and ref_col:
                    return ("r", fk_col, ref_col)

        # Naming-pattern fallback when explicit FK metadata is absent.
        left_cols = [str(col.get("name", "")) for col in tables.get(left_table, {}).get("columns", [])]
        right_cols = [str(col.get("name", "")) for col in tables.get(right_table, {}).get("columns", [])]

        right_singular = right_table[:-1] if right_table.endswith("s") else right_table
        left_singular = left_table[:-1] if left_table.endswith("s") else left_table

        right_id_col = self._find_table_id_column(metadata, right_table)
        left_id_col = self._find_table_id_column(metadata, left_table)

        for col in left_cols:
            c = col.lower()
            if c == f"{right_singular.lower()}_id" or c == right_id_col.lower():
                return ("l", col, right_id_col)
            if right_singular.lower() in c and c.endswith("id"):
                return ("l", col, right_id_col)

        for col in right_cols:
            c = col.lower()
            if c == f"{left_singular.lower()}_id" or c == left_id_col.lower():
                return ("r", col, left_id_col)
            if left_singular.lower() in c and c.endswith("id"):
                return ("r", col, left_id_col)

        return None

    def _pick_grouping_id_column(self, metadata: DatabaseMetadata, table: str) -> str:
        table_info = metadata.schema.get("tables", {}).get(table, {})
        cols = [str(col.get("name", "")) for col in table_info.get("columns", [])]
        id_col = self._find_table_id_column(metadata, table)
        for col in cols:
            c = col.lower()
            if c.endswith("_id") and c != id_col.lower():
                return col
        return id_col

    def _find_join_path(
        self,
        metadata: DatabaseMetadata,
        source_table: str,
        target_table: str,
        max_hops: int = 3,
    ) -> list[tuple[str, str, str, str]]:
        # Each step: (left_table, right_table, left_col, right_col)
        if source_table == target_table:
            return []

        tables = list(metadata.schema.get("tables", {}).keys())
        queue: list[tuple[str, list[tuple[str, str, str, str]], set[str]]] = [(source_table, [], {source_table})]

        while queue:
            current, path, seen = queue.pop(0)
            if len(path) >= max_hops:
                continue

            for nxt in tables:
                if nxt in seen or nxt == current:
                    continue
                edge = self._find_join_edge(metadata, current, nxt)
                if not edge:
                    continue

                side, fk_col, ref_col = edge
                if side == "l":
                    step = (current, nxt, fk_col, ref_col)
                else:
                    step = (current, nxt, ref_col, fk_col)

                new_path = path + [step]
                if nxt == target_table:
                    return new_path

                queue.append((nxt, new_path, seen | {nxt}))

        return []

    def _postprocess_sql(
        self,
        question: str,
        sql: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str:
        if not sql:
            return sql

        # Remove parameterized WHERE clauses (? placeholders) when no specific value
        if " = ?" in sql or " > ?" in sql or " < ?" in sql or " IN ?" in sql:
            sql = re.sub(r"\s+WHERE\s+\w+\s*(=|>|<|IN)\s*\?", "", sql, flags=re.IGNORECASE)

        q = question.lower()
        s = sql.lower()
        table = self._extract_table_from_sql(sql)

        # fix missing threshold comparison (e.g., "older than 20")
        threshold_match = re.search(r"(older|younger|higher|lower|greater|less|more|fewer)\s+than\s+(\d+)", q)
        if threshold_match and " where " not in s:
            operator, threshold_val = threshold_match.groups()
            if any(k in operator for k in ["older", "greater", "higher", "more"]):
                op = ">"
            else:
                op = "<"
            
            # Determine the column name from the question
            column_name = ""
            if "age" in q:
                column_name = self._pick_column_by_keywords(metadata, table, ["age"]) or ""
            elif "score" in q or "grade" in q:
                column_name = self._pick_column_by_keywords(metadata, table, ["score", "grade"]) or ""
            elif "salary" in q:
                column_name = self._pick_column_by_keywords(metadata, table, ["salary"]) or ""
            
            if column_name:
                if " where " in s:
                    sql = f"{sql} AND {column_name} {op} {threshold_val}"
                else:
                    sql = f"{sql} WHERE {column_name} {op} {threshold_val}"

        # fix missing WHERE for active
        if "active" in q and " where " not in f" {s} ":
            target_table = table or self._pick_primary_table(metadata, context)
            active_col = self._pick_column_by_keywords(
                metadata,
                target_table,
                ["is_active", "active", "status", "state", "enabled", "flag"],
            )
            active_condition = "is_active = 1"
            if active_col:
                if any(k in active_col.lower() for k in ["status", "state"]):
                    active_condition = f"LOWER({active_col}) IN ('active', 'current')"
                else:
                    active_condition = f"{active_col} = 1"
            sql = f"{sql} WHERE {active_condition}"
            s = sql.lower()

        # fix missing COUNT
        if "count" in q and "count(" not in s:
            target_table = table or self._pick_primary_table(metadata, context)
            if target_table:
                where_match = re.search(r"(?i)\bwhere\b.+", sql)
                if where_match:
                    sql = f"SELECT COUNT(*) FROM {target_table} {where_match.group(0)}"
                else:
                    sql = f"SELECT COUNT(*) FROM {target_table}"

        # fix advisor intent: ensure related names via join when possible
        if "advisor" in q and " join " not in sql.lower():
            base_table = self._pick_table_from_question(question, metadata, context)
            advisor_table = ""
            for candidate in metadata.schema.get("tables", {}).keys():
                candidate_lower = candidate.lower()
                if any(k in candidate_lower for k in ["advisor", "instructor", "teacher", "mentor", "faculty"]):
                    advisor_table = candidate
                    break

            if base_table and advisor_table:
                join_edge = self._find_join_edge(metadata, base_table, advisor_table)
                if join_edge:
                    left_name = self._pick_name_like_column(metadata, base_table)
                    right_name = self._pick_name_like_column(metadata, advisor_table)
                    if left_name and right_name:
                        edge_side, fk_col, ref_col = join_edge
                        if edge_side == "l":
                            sql = (
                                f"SELECT l.{left_name}, r.{right_name} "
                                f"FROM {base_table} l JOIN {advisor_table} r ON l.{fk_col} = r.{ref_col}"
                            )
                        else:
                            sql = (
                                f"SELECT l.{left_name}, r.{right_name} "
                                f"FROM {base_table} l JOIN {advisor_table} r ON r.{fk_col} = l.{ref_col}"
                            )

        # fix grade intent: avoid id-only projections when a metric column exists
        select_match = re.search(r"(?is)\bselect\b\s+(.*?)\s+\bfrom\b", sql)
        select_text = select_match.group(1).lower() if select_match else ""
        asks_aggregate_or_rank = any(token in q for token in ["average", "avg", "mean", "count", "maximum", "max", "highest", "top", "rank", "second"])
        grade_detail_request = ("grade" in q and not asks_aggregate_or_rank)
        if grade_detail_request and not any(token in select_text for token in ["grade", "score", "rating"]):
            metric = self._find_table_with_metric_keyword(metadata, ["grade", "score", "rating"])
            if metric:
                metric_table, metric_col = metric
                base_table = self._pick_table_from_question(question, metadata, context)
                if base_table and base_table != metric_table:
                    join_edge = self._find_join_edge(metadata, base_table, metric_table)
                    if join_edge:
                        left_name = self._pick_name_like_column(metadata, base_table)
                        edge_side, fk_col, ref_col = join_edge
                        if edge_side == "l":
                            sql = (
                                f"SELECT l.{left_name}, r.{metric_col} "
                                f"FROM {base_table} l JOIN {metric_table} r ON l.{fk_col} = r.{ref_col}"
                            )
                        else:
                            sql = (
                                f"SELECT l.{left_name}, r.{metric_col} "
                                f"FROM {base_table} l JOIN {metric_table} r ON r.{fk_col} = l.{ref_col}"
                            )
                else:
                    sql = f"SELECT {metric_col} FROM {metric_table}"

        # fix maximum-grade intent to explicitly use MAX
        asks_max = any(token in q for token in ["maximum", "max", "highest", "top"])
        plain_max_grade = (
            asks_max
            and "grade" in q
            and not any(token in q for token in ["average", "avg", "mean", "second", "each", "per", "rank", "top 5"])
        )
        if plain_max_grade and "max(" not in sql.lower():
            metric = self._find_table_with_metric_keyword(metadata, ["grade", "score", "rating"])
            if metric:
                metric_table, metric_col = metric
                sql = f"SELECT MAX({metric_col}) AS max_value FROM {metric_table}"

        # fix relation phrases like "associated with" by forcing schema-based join when possible
        if any(phrase in q for phrase in ["associated with", "related to", "involving"]) and " join " not in sql.lower():
            mentioned = self._mentioned_tables_in_question(question, metadata)
            if len(mentioned) >= 2:
                left_table, right_table = mentioned[0], mentioned[1]
                left_name = self._pick_name_like_column(metadata, left_table)
                right_name = self._pick_name_like_column(metadata, right_table)
                left_id = self._find_table_id_column(metadata, left_table)
                right_id = self._find_table_id_column(metadata, right_table)

                edge = self._find_join_edge(metadata, left_table, right_table)
                if edge and left_name and right_name:
                    side, fk_col, ref_col = edge
                    if side == "l":
                        sql = f"SELECT l.{left_name}, r.{right_name} FROM {left_table} l JOIN {right_table} r ON l.{fk_col} = r.{ref_col}"
                    else:
                        sql = f"SELECT l.{left_name}, r.{right_name} FROM {left_table} l JOIN {right_table} r ON r.{fk_col} = l.{ref_col}"
                else:
                    bridge = self._find_bridge_table(metadata, left_table, right_table)
                    if bridge and left_name and right_name and left_id and right_id:
                        bridge_table, fk_left, fk_right = bridge
                        sql = (
                            f"SELECT l.{left_name}, r.{right_name} "
                            f"FROM {left_table} l "
                            f"JOIN {bridge_table} b ON b.{fk_left} = l.{left_id} "
                            f"JOIN {right_table} r ON b.{fk_right} = r.{right_id}"
                        )

        # fix latest intent: require temporal max subquery when question asks latest/most recent
        asks_latest = any(token in q for token in ["latest", "most recent", "newest"])
        if asks_latest and "max(" not in sql.lower() and "order by" not in sql.lower():
            # Look for a table with the metric mentioned in the question (performance, rating, grade, etc.)
            metric_table = ""
            temporal_col = ""
            metric_col = ""
            
            # First, find a table that contains a metric column matching the question keywords
            for table in metadata.schema.get("tables", {}).keys():
                metric_keywords = ["performance", "rating", "score", "grade", "feedback", "review"]
                table_lower = table.lower()
                if any(keyword in table_lower or table_lower in ("ratings", "grades", "performances") for keyword in metric_keywords):
                    metric_col = self._pick_metric_column(metadata, table)
                    temporal_col = self._pick_temporal_column(metadata, table)
                    if metric_col and temporal_col:
                        metric_table = table
                        break
            
            # If no metric table found with temporal column, skip this correction
            if metric_table and temporal_col and metric_col:
                fk_col = self._pick_grouping_id_column(metadata, metric_table)
                if fk_col:
                    sql = (
                        f"SELECT t1.{fk_col}, t1.{metric_col} "
                        f"FROM {metric_table} t1 "
                        f"WHERE t1.{temporal_col} = ("
                        f"SELECT MAX(t2.{temporal_col}) FROM {metric_table} t2 WHERE t2.{fk_col} = t1.{fk_col}"
                        ")"
                    )

        # fix aggregate comparison intent like "total ... exceed ..."
        asks_exceed_total = any(token in q for token in ["exceed", "greater than", "more than"]) and any(token in q for token in ["total", "sum"])
        if asks_exceed_total and "sum(" not in sql.lower():
            sum_table = ""
            threshold_table = ""
            sum_col = ""
            threshold_col = ""
            
            # Prefer explicitly mentioned tables from the question
            mentioned_tables = self._mentioned_tables_in_question(question, metadata)
            all_tables = mentioned_tables + list(metadata.schema.get("tables", {}).keys())
            
            for t in all_tables:
                if not sum_col:
                    # Prefer "transaction" or "transaction_related" keywords for sum_table
                    c = self._pick_column_by_keywords(metadata, t, ["amount", "hours", "quantity", "value", "total"])
                    if c and ("transaction" in t.lower() or "amount" in c.lower()):
                        sum_table, sum_col = t, c
                if not threshold_col:
                    c = self._pick_column_by_keywords(metadata, t, ["budget", "limit", "target", "cap", "quota"])
                    if c:
                        threshold_table, threshold_col = t, c

            if sum_table and threshold_table and sum_col and threshold_col:
                path = self._find_join_path(metadata, sum_table, threshold_table, max_hops=2)
                if path:
                    alias_map: dict[str, str] = {}
                    alias_seq = iter(["s", "j", "k", "p"])
                    def alias_for(table_name: str) -> str:
                        if table_name not in alias_map:
                            alias_map[table_name] = next(alias_seq)
                        return alias_map[table_name]

                    first_left, first_right, first_lc, first_rc = path[0]
                    from_clause = f"FROM {first_left} {alias_for(first_left)} JOIN {first_right} {alias_for(first_right)} ON {alias_for(first_left)}.{first_lc} = {alias_for(first_right)}.{first_rc}"
                    for step in path[1:]:
                        left_t, right_t, left_c, right_c = step
                        from_clause += f" JOIN {right_t} {alias_for(right_t)} ON {alias_for(left_t)}.{left_c} = {alias_for(right_t)}.{right_c}"

                    threshold_id = self._find_table_id_column(metadata, threshold_table)
                    sql = (
                        f"SELECT {alias_for(threshold_table)}.{threshold_id} "
                        f"{from_clause} "
                        f"GROUP BY {alias_for(threshold_table)}.{threshold_id} "
                        f"HAVING SUM({alias_for(sum_table)}.{sum_col}) > MAX({alias_for(threshold_table)}.{threshold_col})"
                    )

        # fix diversity intent like "more than one country"
        asks_diversity = any(token in q for token in ["more than one", "more than 1"]) and "country" in q
        if asks_diversity and "count(distinct" not in sql.lower():
            mentioned = self._mentioned_tables_in_question(question, metadata)
            if mentioned:
                base = mentioned[0]
                country_table = ""
                
                # Find the table with country information; prefer "clients" table based on question keywords
                for t in metadata.schema.get("tables", {}).keys():
                    if self._pick_country_like_column(metadata, t):
                        # Prefer clients table if question mentions clients/projects
                        if ("client" in q or "project" in q) and "client" in t.lower():
                            country_table = t
                            break
                        elif not country_table:
                            country_table = t
                
                if country_table:
                    country_col = self._pick_country_like_column(metadata, country_table)
                    path = self._find_join_path(metadata, base, country_table, max_hops=3)
                    if path and country_col:
                        alias_map: dict[str, str] = {}
                        alias_seq = iter(["b", "j", "k", "c"])
                        def alias_for(table_name: str) -> str:
                            if table_name not in alias_map:
                                alias_map[table_name] = next(alias_seq)
                            return alias_map[table_name]

                        first_left, first_right, first_lc, first_rc = path[0]
                        from_clause = f"FROM {first_left} {alias_for(first_left)} JOIN {first_right} {alias_for(first_right)} ON {alias_for(first_left)}.{first_lc} = {alias_for(first_right)}.{first_rc}"
                        for step in path[1:]:
                            left_t, right_t, left_c, right_c = step
                            from_clause += f" JOIN {right_t} {alias_for(right_t)} ON {alias_for(left_t)}.{left_c} = {alias_for(right_t)}.{right_c}"

                        base_id = self._find_table_id_column(metadata, base)
                        sql = (
                            f"SELECT {alias_for(base)}.{base_id} "
                            f"{from_clause} "
                            f"GROUP BY {alias_for(base)}.{base_id} HAVING COUNT(DISTINCT {alias_for(country_table)}.{country_col}) > 1"
                        )

        return sql

    def _generate_rule_fallback_sql(
        self,
        question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        q = question.lower()
        table = self._pick_table_from_question(question, metadata, context)
        if not table:
            return None

        where_parts: list[str] = []

        if "active" in q:
            active_col = self._pick_column_by_keywords(
                metadata,
                table,
                ["is_active", "active", "status", "state", "enabled", "flag"],
            )
            if active_col:
                if any(k in active_col.lower() for k in ["status", "state"]):
                    where_parts.append(f"LOWER({active_col}) IN ('active', 'current')")
                else:
                    where_parts.append(f"{active_col} = 1")

        # Handle age threshold (older/younger than N) before temporal year filters
        threshold_match = re.search(r"(older|younger|higher|lower)\s+than\s+(\d+)", q)
        if threshold_match:
            operator_text, threshold_val = threshold_match.groups()
            if any(k in operator_text for k in ["older", "higher"]):
                op = ">"
            else:
                op = "<"
            
            age_col = self._pick_column_by_keywords(metadata, table, ["age"])
            if age_col:
                where_parts.append(f"{age_col} {op} {threshold_val}")
        elif any(k in q for k in ["year", "after", "before", " in ", "since", "from"]):
            # Only apply temporal year filter if no age threshold
            year_col = self._pick_column_by_keywords(metadata, table, ["enrollment_year", "year", "date", "semester"])
            if year_col:
                year_match = re.search(r"\b(19|20)\d{2}\b", q)
                if year_match:
                    year_val = year_match.group(0)
                    if "after" in q:
                        where_parts.append(f"{year_col} > {year_val}")
                    elif "before" in q:
                        where_parts.append(f"{year_col} < {year_val}")
                    else:
                        where_parts.append(f"{year_col} = {year_val}")
                else:
                    where_parts.append(f"{year_col} IS NOT NULL")

        # If year intent exists but chosen table has no year/date column,
        # switch to a better table that does.
        if any(k in q for k in ["year", "after", "before", " in ", "since", "from"]) and not where_parts:
            year_match = re.search(r"\b(19|20)\d{2}\b", q)
            for candidate_table in metadata.schema.get("tables", {}).keys():
                candidate_year_col = self._pick_column_by_keywords(
                    metadata,
                    candidate_table,
                    ["enrollment_year", "year", "date", "semester"],
                )
                if not candidate_year_col:
                    continue

                table = candidate_table
                if year_match:
                    year_val = year_match.group(0)
                    if "after" in q:
                        where_parts.append(f"{candidate_year_col} > {year_val}")
                    elif "before" in q:
                        where_parts.append(f"{candidate_year_col} < {year_val}")
                    else:
                        where_parts.append(f"{candidate_year_col} = {year_val}")
                else:
                    where_parts.append(f"{candidate_year_col} IS NOT NULL")
                break

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        asks_grouping = bool(re.search(r"\beach\b|\bper\b|for each|in each|group by", q))
        asks_count = any(k in q for k in ["count", "how many", "total", "number of"])
        asks_average = any(k in q for k in ["average", "avg", "mean"])

        def pick_group_col(target_table: str) -> str:
            # Prefer department if question mentions it, otherwise use other categorical columns
            if "department" in q:
                dept_col = self._pick_column_by_keywords(metadata, target_table, ["department"])
                if dept_col:
                    return dept_col
            
            # Prefer question-aligned categorical columns, then common grouping fields.
            group_col = self._pick_column_by_keywords(
                metadata,
                target_table,
                ["department", "semester", "term", "category", "type", "status", "title", "label"],
            )
            if group_col:
                return group_col
            # Only fallback to "name" if no categorical columns found
            return self._pick_column_by_keywords(metadata, target_table, ["name"]) or ""

        # COUNT fallback
        if asks_count:
            if asks_grouping:
                group_col = pick_group_col(table)
                if group_col:
                    return f"SELECT {group_col}, COUNT(*) AS total_count FROM {table}{where_clause} GROUP BY {group_col}"
            return f"SELECT COUNT(*) FROM {table}{where_clause}"

        # AVG fallback
        if asks_average:
            metric_keywords = ["score", "grade", "salary", "amount", "gpa", "credits", "hours", "age", "price", "cost", "rating"]
            metric_col = self._pick_column_by_keywords(metadata, table, metric_keywords)

            # If target table does not have a metric column, pick any table that has one.
            # But prefer a table that matches the question's intent (e.g., students for "age", employees for "salary")
            if not metric_col:
                preference_order = []
                
                # Special handling for "age" - prefer enrollment_year on students if available
                if any(k in q for k in ["age", "student"]):
                    for t in metadata.schema.get("tables", {}).keys():
                        if "student" in t.lower():
                            age_col = self._pick_column_by_keywords(metadata, t, ["age", "enrollment_year"])
                            if age_col:
                                table = t
                                metric_col = age_col
                                break
                
                if not metric_col and any(k in q for k in ["salary", "employee"]):
                    preference_order = [t for t in metadata.schema.get("tables", {}).keys() if "employee" in t.lower()]
                
                # Try preferred tables first
                if not metric_col:
                    for candidate_table in preference_order + list(metadata.schema.get("tables", {}).keys()):
                        candidate_metric = self._pick_column_by_keywords(metadata, candidate_table, metric_keywords)
                        if candidate_metric:
                            table = candidate_table
                            metric_col = candidate_metric
                            break

            if metric_col:
                if asks_grouping:
                    group_col = pick_group_col(table)
                    if group_col:
                        return f"SELECT {group_col}, AVG({metric_col}) AS average_value FROM {table}{where_clause} GROUP BY {group_col}"
                return f"SELECT AVG({metric_col}) AS average_value FROM {table}{where_clause}"

        # Projection fallback: avoid SELECT * when question asks for specific attributes.
        projected: list[str] = []
        if re.search(r"\bnames?\b", q):
            name_col = self._pick_column_by_keywords(metadata, table, ["name"]) or self._pick_column_by_keywords(
                metadata,
                table,
                ["title", "label"],
            )
            if name_col:
                projected.append(name_col)
        if re.search(r"\bcredits\b", q):
            credits_col = self._pick_column_by_keywords(metadata, table, ["credits", "credit"])
            if credits_col and credits_col not in projected:
                projected.append(credits_col)

        if projected:
            return f"SELECT {', '.join(projected)} FROM {table}{where_clause}"

        if where_clause:
            return f"SELECT * FROM {table}{where_clause}"

        return None

    # -----------------------------
    def generate(self, question: str, metadata: DatabaseMetadata, context: RetrievalContext) -> str:
        logger.info(f"Question: {question}")

        if not self._is_question_relevant(question, metadata):
            logger.warning("[VALIDATION] Question not relevant to database")
            return "INVALID_QUERY"

        # -----------------------------
        # SIMPLE QUERY: prefer Flan-only LLM path (avoid hardcoded SQL)
        # -----------------------------
        is_simple = self._handle_simple_query(question, metadata, context)
        bundles = self._load_bundles()

        if is_simple:
            # Try Flan-family models first for simple queries
            for bundle in bundles:
                if bundle.model_key == "mistral_7b":
                    continue
                try:
                    sql = self._generate_with_bundle(bundle, question, metadata, context)
                    if sql:
                        logger.info("[SIMPLE PATH] Using Flan-only result")
                        return sql
                except Exception as e:
                    logger.error(f"{bundle.model_key} failed on simple query: {e}")

            logger.debug("[SIMPLE PATH] Flan models produced no acceptable SQL; falling back to complex path")

        # Default complex path: try all bundles (Flan first, then Mistral)
        # Ensure Flan models are attempted before Mistral for best single-model coverage
        bundles_sorted = sorted(bundles, key=lambda b: 0 if b.model_key != "mistral_7b" else 1)
        for bundle in bundles_sorted:
            try:
                sql = self._generate_with_bundle(bundle, question, metadata, context)
                if sql:
                    return sql
            except Exception as e:
                logger.error(f"{bundle.model_key} failed: {e}")

        rule_fallback_sql = self._generate_rule_fallback_sql(question, metadata, context)
        if rule_fallback_sql and self._looks_like_sql(rule_fallback_sql) and self._is_sql_acceptable(question, rule_fallback_sql, metadata):
            return rule_fallback_sql

        fallback_sql = self.fallback_generator.generate(question, metadata, context)
        fallback_sql = self._postprocess_sql(question, fallback_sql, metadata, context)
        if self._looks_like_sql(fallback_sql):
            if not self._is_sql_acceptable(question, fallback_sql, metadata):
                logger.warning("[VALIDATION] Fallback SQL failed strict acceptability checks; returning for validator/semantic guard")
            return fallback_sql

        return ""

    def repair(
        self,
        question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
        error_message: str,
        previous_sql: str,
    ) -> str:
        if not self._is_question_relevant(question, metadata):
            logger.warning("[VALIDATION] Question not relevant to database")
            return "INVALID_QUERY"

        bundles = self._load_bundles()
        for bundle in bundles:
            try:
                sql = self._generate_with_bundle(bundle, question, metadata, context)
                if sql:
                    return sql
            except Exception as e:
                logger.error(f"{bundle.model_key} repair failed: {e}")

        fallback_sql = self.fallback_generator.repair(
            question,
            metadata,
            context,
            error_message,
            previous_sql,
        )
        fallback_sql = self._postprocess_sql(question, fallback_sql, metadata, context)
        if self._looks_like_sql(fallback_sql):
            if not self._is_sql_acceptable(question, fallback_sql, metadata):
                logger.warning("[VALIDATION] Fallback repair SQL failed strict acceptability checks; returning for validator/semantic guard")
            return fallback_sql
        return ""

    # -----------------------------
    def _generate_with_bundle(self, bundle, question, metadata, context):
        schema_lines = self._build_schema_lines(metadata, context)

        if bundle.model_key == "mistral_7b":
            system_text, user_text = build_mistral_model_prompt(
                question=question,
                schema_text="; ".join(schema_lines),
                table_hints_text="",
                repair=False,
                error_message="",
                previous_sql="",
                examples_text="",
            )
            # Mistral requires proper chat format: [INST] system + user [/INST]
            prompt = f"[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n{user_text} [/INST]"
            generation_kwargs = mistral_model_generation_kwargs(160)
        else:
            prompt = build_flan_t5_model_prompt(
                question=question,
                schema_text="; ".join(schema_lines),
                table_hints_text="",
                repair=False,
                error_message="",
                previous_sql="",
                examples_text="",
            )
            generation_kwargs = flan_t5_model_generation_kwargs(64)

        tokenizer = bundle.tokenizer
        model = bundle.model

        if bundle.model_key == "mistral_7b" and torch.cuda.is_available():
            try:
                if getattr(model, "device", None) is None or str(model.device) != "cuda:0":
                    model = model.to("cuda")
                    bundle.model = model
                    logger.info("[MISTRAL] Moved model to CUDA")
            except Exception as exc:
                logger.warning(f"[MISTRAL] Could not move model to CUDA: {exc}")

        # Tokenize with model-specific max_length: Mistral supports longer sequences
        if bundle.model_key == "mistral_7b":
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        else:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(**inputs, **generation_kwargs)
        if bundle.model_key == "mistral_7b":
            prompt_len = int(inputs["input_ids"].shape[1])
            output_len = int(outputs[0].shape[0])
            generated_ids = outputs[0][prompt_len:]
            # For Mistral, decode without special tokens stripped first
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
            # Clean up EOS token and other markers
            decoded = decoded.replace('</s>', '').strip()
        else:
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Initialize last-trace storage for debugging/provenance
        try:
            self._last_trace = {
                "model": bundle.model_key,
                "decoded": decoded,
                "extracted": "",
                "postprocessed": "",
            }
        except Exception:
            self._last_trace = None

        # Log raw LLM output to terminal prominently
        try:
            compact = re.sub(r"\s+", " ", decoded).strip()
            preview = compact[:500] + ("..." if len(compact) > 500 else "")
            logger.info(f"[RAW_OUTPUT] model={bundle.model_key} decoded='{preview}'")
        except Exception:
            pass

        # Clean raw output before extraction (strip dataset junk tokens)
        cleaned = self._clean_raw_output(decoded)
        if cleaned != decoded:
            logger.debug(f"[CLEANED] Removed junk tokens. Before: {decoded[:100]}... After: {cleaned[:100]}...")

        sql = self._extract_sql(cleaned)
        if getattr(self, "_last_trace", None) is not None:
            try:
                self._last_trace["extracted"] = sql
            except Exception:
                pass

        # 🚨 BLOCK GARBAGE
        if self._is_garbage_output(cleaned):
            logger.warning(f"[GARBAGE] {bundle.model_key} output contains junk tokens")
            if getattr(self, "_last_trace", None) is not None:
                self._last_trace["extracted"] = ""
                self._last_trace["postprocessed"] = ""
            return ""

        sql = self._postprocess_sql(question, sql, metadata, context)
        if getattr(self, "_last_trace", None) is not None:
            try:
                self._last_trace["postprocessed"] = sql
            except Exception:
                pass

        if not self._looks_like_sql(sql):
            logger.warning(f"[VALIDATION] {bundle.model_key} SQL extraction failed (not valid SQL syntax)")
            return ""

        if not self._is_sql_acceptable(question, sql, metadata):
            logger.warning(f"[VALIDATION] {bundle.model_key} SQL failed acceptability checks")
            logger.debug(f"  Question: {question[:80]}...")
            logger.debug(f"  Raw extracted: {sql[:150]}...")
            return ""

        # Log successful generation with full pipeline trace
        logger.info(f"[ACCEPTED] {bundle.model_key} ✓ SQL passed all checks")
        logger.debug(f"  [TRACE] Raw: {decoded[:200]}")
        logger.debug(f"  [TRACE] Extracted: {sql[:150]}")
        logger.debug(f"  [TRACE] Final SQL: {sql}")
        return sql

    # -----------------------------
    # 🔥 STRICT SQL EXTRACTION
    # -----------------------------
    def _extract_sql(self, decoded: str) -> str:
        text = decoded.strip()

        match = re.search(r"(?i)\b(select|with)\b.+", text)
        if not match:
            return ""

        sql = match.group(0)
        sql = re.split(r"'agg'|'conds'|'sel'|column_index", sql)[0]

        return sql.split(";")[0].strip()

    # -----------------------------
    def _looks_like_sql(self, sql: str) -> bool:
        if not sql:
            return False

        lowered = sql.lower()

        if any(token in lowered for token in ["'sel'", "'agg'", "'conds'"]):
            return False

        if not lowered.startswith(("select", "with")):
            return False

        try:
            sqlglot.parse_one(sql, read="sqlite")
            return True
        except Exception:
            return False

    # -----------------------------
    def _build_schema_lines(self, metadata, context):
        tables = metadata.schema.get("tables", {})
        relevant = context.relevant_tables or list(tables.keys())[:3]

        lines = []
        for t in relevant:
            cols = [c["name"] for c in tables[t]["columns"]][:6]
            lines.append(f"{t}({', '.join(cols)})")

        return lines

    # -----------------------------
    def _load_bundles(self):
        if self._bundles:
            return self._bundles

        project_root = Path(__file__).resolve().parents[2]
        setup_adapters(project_root)

        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[LLM_SQL_GENERATOR] Loading local model on device: {device}")

        def infer_model_key(path: str) -> str:
            p = Path(path)

            # 1) Detect via config.json model_type/architectures
            try:
                cfg_path = p / "config.json"
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    model_type = str(cfg.get("model_type", "")).lower()
                    arches = [str(a).lower() for a in cfg.get("architectures", [])]
                    if model_type == "mistral" or any("mistral" in a for a in arches):
                        return "mistral_7b"
            except Exception:
                pass

            # 2) Detect via adapter_config base model name
            try:
                adapter_cfg_path = p / "adapter_config.json"
                if adapter_cfg_path.exists():
                    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
                    base_name = str(adapter_cfg.get("base_model_name_or_path", "")).lower()
                    if "mistral" in base_name:
                        return "mistral_7b"
            except Exception:
                pass

            # 3) Fallback to path heuristic
            return "mistral_7b" if "mistral" in path.lower() else "flan_t5"

        bundles = []
        for path in self.model_paths:
            model_key = infer_model_key(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
            if model_key == "mistral_7b":
                if not torch.cuda.is_available():
                    logger.warning("[LLM_SQL_GENERATOR] Mistral 7B requires CUDA GPU, skipping...")
                    continue
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    device_map="auto",
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                )
                family = "causal"
            else:
                adapter_config_path = Path(path) / "adapter_config.json"
                if adapter_config_path.exists():
                    adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                    base_model_name = adapter_cfg.get("base_model_name_or_path", "google/flan-t5-small")
                else:
                    base_model_name = "google/flan-t5-small"

                base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                model = PeftModel.from_pretrained(base_model, path)
                family = "seq2seq"

            if model_key != "mistral_7b":
                model = model.to(device)
            model.eval()

            bundles.append(_ModelBundle(
                tokenizer=tokenizer,
                model=model,
                family=family,
                model_path=path,
                model_key=model_key,
            ))

        self._bundles = bundles
        return bundles