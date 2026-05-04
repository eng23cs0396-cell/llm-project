from __future__ import annotations

import re


def _classify_question(question: str) -> set[str]:
    q = question.lower()
    tags = set()

    # Refined keywords focused on common patterns in education/company databases
    join_kw = ["with their", "along with", "working in", "belong", "under", "part of", "assigned to", "enrolled in", "teach", "taught by", "advisor", "whose"]
    agg_kw = ["count", "how many", "total", "sum", "average", "avg", "per ", "each", "most", "least", "highest", "lowest", "maximum", "minimum", "more than", "fewer than", "group", "number of", "less than", "greater than"]
    date_kw = ["date", "year", "month", "day", "since", "before", "after", "hired", "joined", "when", "duration", "tenure", "age", "elapsed", "latest", "recent", "oldest", "newest"]
    rank_kw = ["rank", "top ", "second", "third", "fourth", "fifth", "nth", "2nd", "3rd", "4th", "offset", "running", "cumulative", "dense", "second highest", "second lowest", "third highest", "third lowest", "higher", "lower"]
    compute_kw = ["percentage", "percent", "%", "ratio", "share", "growth", "increase", "difference", "above average", "below average", "running total"]
    filter_kw = ["greater than", "less than", "more than", "fewer than", "at least", "at most", "equal", "not equal", "older than", "younger than"]

    if any(re.search(k, q) for k in join_kw):
        tags.add("join")
    if any(k in q for k in agg_kw):
        tags.add("aggregate")
    if any(k in q for k in date_kw):
        tags.add("date")
    if any(k in q for k in rank_kw):
        tags.add("rank")
    if any(k in q for k in compute_kw):
        tags.add("compute")
    if any(k in q for k in filter_kw):
        tags.add("filter")

    return tags


def build_prompt(
    *,
    question: str,
    schema_text: str,
    table_hints_text: str = "",
    repair: bool,
    error_message: str,
    previous_sql: str,
    examples_text: str = "",
) -> tuple[str, str]:
    tags = _classify_question(question)
    system_text = (
        "You are an expert SQLite text-to-SQL assistant. "
        "Return exactly one valid SQLite SELECT/WITH statement and nothing else. "
        "CRITICAL: Do NOT output dataset-specific tokens like 'sel', 'agg', 'conds', column_index, operator_index. "
        "Output only clean SQL code."
    )

    hints: list[str] = [
        "=== CORE RULES ===",
        "• Output exactly one SQLite SELECT/WITH query.",
        "• Output only SQL; no prose, no markdown.",
        "• Use only schema tables/columns.",
        "• Keep all numeric values and phrasing exactly from question.",
        "",
        "=== COMPARISON OPERATORS ===",
        "• 'after X' (temporal) → WHERE column > X",
        "• 'before X' (temporal) → WHERE column < X",
        "• 'between X and Y' → BETWEEN X AND Y",
        "• 'greater than', 'more than' → >",
        "• 'less than', 'fewer than' → <",
        "• 'at least' → >=  |  'at most' → <=",
        "• 'not equal', 'different' → !=  |  'in X' → =",
        "",
        "=== AGGREGATION (VALUE vs ENTITY) ===",
        "• VALUE queries (what is the highest/lowest/average): MAX/MIN/AVG/SUM/COUNT",
        "• ENTITY queries (which person/who): ORDER BY + LIMIT, NO aggregation",
        "• 'highest', 'max', 'top value' → MAX()  |  'lowest', 'min' → MIN()",
        "• 'average', 'mean' → AVG()  |  'total', 'sum' → SUM()  |  'count', 'how many' → COUNT()",
        "",
        "=== RANKING ===",
        "• 'top N' → ORDER BY DESC LIMIT N",
        "• 'lowest N' → ORDER BY ASC LIMIT N",
        "• 'Nth highest/lowest' → LIMIT 1 OFFSET (N-1)",
        "• 'second highest' → ORDER BY DESC LIMIT 1 OFFSET 1",
        "",
        "=== GROUPING ===",
        "• 'per', 'each', 'for each', 'grouped by' → GROUP BY",
        "• CRITICAL: ALL non-aggregated columns MUST be in GROUP BY clause",
        "• WHERE applies BEFORE GROUP BY | HAVING applies AFTER",
        "",
        "=== NULL HANDLING ===",
        "• NEVER use = NULL or != NULL",
        "• Use IS NULL / IS NOT NULL only",
        "• 'missing', 'empty', 'no value', 'unassigned' → IS NULL",
        "• 'has value', 'assigned', 'active' → IS NOT NULL",
        "",
        "=== JOINS ===",
        "• Join when data needed from MULTIPLE tables",
        "• Use foreign key relationships from schema (student_id, course_id, etc.)",
        "• INNER JOIN for required relationships | LEFT JOIN for optional",
        "• Connect via id columns, not name=name joins",
        "",
        "=== PERCENTAGES ===",
        "• 'percentage of X' → (COUNT(X)*100.0/COUNT(total))",
        "• ALWAYS use float division: 100.0, not 100",
        "• Identify denominator carefully: total vs filtered subset",
    ]
    
    if "join" in tags:
        hints.append("• This requires joining tables. Use explicit JOINs with real foreign keys from the schema.")
    if "aggregate" in tags:
        hints.append("• This requires aggregation. Use COUNT/SUM/AVG/MAX/MIN with GROUP BY as needed.")
    if "rank" in tags:
        hints.append("• This requires ranking/ordering. Use ORDER BY with LIMIT/OFFSET to get top or specific positions.")
    if "date" in tags:
        hints.append("• This involves dates or temporal data. Use date-aware column selection and filtering.")
    if "compute" in tags:
        hints.append("• This requires computed values. Calculate using SQL arithmetic operations or aggregation.")
    if "filter" in tags:
        hints.append("• This requires filtering by conditions. Apply WHERE clauses with correct comparators.")

    hint_block = "\n".join(hints)
    table_hints_block = f"\nRelevant tables/columns:\n{table_hints_text}\n" if table_hints_text.strip() else ""
    examples_block = (
        "\n=== REFERENCE PATTERNS (ADAPT; DO NOT COPY LITERALLY) ===\n"
        f"{examples_text}\n"
        if examples_text.strip()
        else ""
    )

    if repair:
        user_text = (
            "Task: Repair the SQL query to fix the validation error.\n"
            f"Question: {question}\n"
            f"Schema: {schema_text}\n"
            f"{table_hints_block}"
            f"Validation Error: {error_message}\n"
            f"Previous SQL: {previous_sql}\n"
            f"\n{hint_block}{examples_block}\n"
            "Fixed SQL:"
        )
    else:
        user_text = (
            "Task: Generate a valid SQL query for the question.\n"
            f"Question: {question}\n"
            f"Schema: {schema_text}\n"
            f"{table_hints_block}"
            f"\n{hint_block}{examples_block}\n"
            "SQL:"
        )

    return system_text, user_text


def generation_kwargs(max_new_tokens: int) -> dict:
    return {
        "max_new_tokens": max(96, min(220, max_new_tokens)),
        "temperature": 0.1,  # Very low temperature for deterministic, high-quality output
        "top_p": 0.95,
        "do_sample": True,
        "use_cache": True,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    }