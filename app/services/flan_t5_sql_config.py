from __future__ import annotations

import re


def _classify_question(question: str) -> set[str]:
    q = question.lower()
    tags = set()

    if any(k in q for k in ["join", "with their", "along with", "assigned to", "belongs to"]):
        tags.add("join")

    if any(k in q for k in ["count", "how many", "total", "number of"]):
        tags.add("count")

    if any(k in q for k in ["average", "avg", "mean"]):
        tags.add("average")

    if any(k in q for k in ["top", "highest", "lowest", "maximum", "minimum"]):
        tags.add("ranking")

    if any(k in q for k in ["greater than", "less than", "more than", "at least", "at most", "after", "before"]):
        tags.add("filter")

    if any(k in q for k in ["each", "per", "group by"]):
        tags.add("group")

    return tags


def _build_hint_line(tags: set[str]) -> str:
    hints = []

    if "count" in tags:
        hints.append("Use COUNT(*)")

    if "average" in tags:
        hints.append("Use AVG(column)")

    if "group" in tags:
        hints.append("Use GROUP BY")

    if "ranking" in tags:
        hints.append("Use ORDER BY ... DESC LIMIT")

    if "join" in tags:
        hints.append("Use JOIN with ON condition")

    if "filter" in tags:
        hints.append("Use WHERE with correct operators (=, >, <, >=, <=)")

    if not hints:
        return ""

    return "Hint: " + "; ".join(hints)


def build_prompt(
    *,
    question: str,
    schema_text: str,
    table_hints_text: str = "",
    repair: bool,
    error_message: str,
    previous_sql: str,
    examples_text: str = "",
) -> str:
    tags = _classify_question(question)
    hint_line = _build_hint_line(tags)

    table_block = f"Tables:\n{table_hints_text}\n" if table_hints_text.strip() else ""

    rules = (
        "- Use only table and column names from schema\n"
        "- Use JOIN only if needed\n"
        "- Use WHERE for conditions\n"
        "- Use GROUP BY for per/each\n"
        "- Use ORDER BY and LIMIT for top/highest\n"
    )

    if repair:
        return (
            "Fix SQL query.\n"
            f"Question: {question}\n"
            f"Schema: {schema_text}\n"
            f"{table_block}"
            f"{hint_line}\n"
            f"Error: {error_message}\n"
            f"SQL: {previous_sql}\n"
            f"{rules}"
            "OUTPUT INSTRUCTIONS:\n"
            "- Return ONLY valid SQLite SQL\n"
            "- Do NOT output: 'sel', 'agg', 'conds', column_index, operator_index\n"
            "- Do NOT include explanations, prose, or markdown\n"
            "- Start with SELECT or WITH\n"
            "Output format:\nSELECT ...\n"
        )

    return (
        "Generate SQL query.\n"
        f"Question: {question}\n"
        f"Schema: {schema_text}\n"
        f"{table_block}"
        f"{hint_line}\n"
        f"{rules}"
        "OUTPUT INSTRUCTIONS:\n"
        "- Return ONLY valid SQLite SQL\n"
        "- Do NOT output: 'sel', 'agg', 'conds', column_index, operator_index\n"
        "- Do NOT include explanations, prose, or markdown\n"
        "- Start with SELECT or WITH\n"
        "Output format:\nSELECT ...\n"
    )


def generation_kwargs(max_new_tokens: int) -> dict:
    return {
        "max_new_tokens": max(64, min(128, max_new_tokens)),
        "num_beams": 4,
        "do_sample": True,
        "temperature": 0.2,  # Lower temperature for more deterministic, clean output
        "top_p": 0.95,
        "use_cache": True,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "length_penalty": 1.0,
        "early_stopping": True,
    }