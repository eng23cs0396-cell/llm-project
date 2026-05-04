from __future__ import annotations

import re

from app.storage import DatabaseMetadata


class SQLSemanticGuard:
    """Question-aware SQL sanity checks without hardcoded question examples."""

    def check(self, question: str, sql: str, metadata: DatabaseMetadata | None = None) -> list[str]:
        issues: list[str] = []
        lowered_q = question.lower()
        lowered_sql = sql.lower()
        sql_tokens = self._tokenize(lowered_sql)

        q_numbers = {int(n) for n in re.findall(r"\b\d+\b", lowered_q)}
        sql_numbers = [int(n) for n in re.findall(r"\b\d+\b", lowered_sql)]

        # If question has numeric constraints (threshold, count), SQL should preserve them.
        # Allow for flexibility when numbers are implicit (e.g., "top 5" might not show 5 in sql if using LIMIT without showing the value)
        if q_numbers and sql_numbers:
            if not any(n in q_numbers for n in sql_numbers):
                # Exception: if question asks for "top N" but SQL uses LIMIT without N visible, allow it
                if not ("limit" in lowered_sql and any(kw in lowered_q for kw in ["top", "highest", "lowest", "first"])):
                    issues.append("Numeric constraint mismatch between question and SQL")

        # Guard against hallucinated giant literals for small-scale business/academic queries.
        if sql_numbers:
            largest_sql_number = max(sql_numbers)
            largest_q_number = max(q_numbers) if q_numbers else 0
            if largest_sql_number >= 1_000_000_000 and largest_q_number < 1_000_000:
                issues.append("Suspiciously large numeric literal in SQL")

        schema_vocab: set[str] = set()
        if metadata is not None:
            for table, info in metadata.schema.get("tables", {}).items():
                schema_vocab.update(self._tokenize(table))
                for column in info.get("columns", []):
                    schema_vocab.update(self._tokenize(str(column.get("name", ""))))

        question_tokens = self._question_tokens(lowered_q)
        salient_schema_terms = [
            token for token in question_tokens if token in schema_vocab and token not in self._stopwords()
        ]

        # If the question clearly names schema entities, the SQL should mention them too.
        # But allow for FK references (e.g., course_id satisfies "courses")
        if metadata is not None and salient_schema_terms:
            missing_terms = [
                term
                for term in salient_schema_terms
                if not term.endswith("id") and not self._term_is_present(term, lowered_sql, sql_tokens)
            ]
            # Only report if we have strong evidence of missing key entities
            if len(missing_terms) >= 1 and len(salient_schema_terms) >= 2:
                issues.append(f"SQL may be missing key table/entity: {', '.join(sorted(set(missing_terms[:2])))}")

        # Comparator alignment checks for common quantitative intents.
        comparator_patterns = [
            (r"older than\s+(\d+)", ">"),
            (r"more than\s+(\d+)", ">"),
            (r"greater than\s+(\d+)", ">"),
            (r"less than\s+(\d+)", "<"),
            (r"younger than\s+(\d+)", "<"),
            (r"at least\s+(\d+)", ">="),
            (r"at most\s+(\d+)", "<="),
        ]

        for pattern, expected_op in comparator_patterns:
            match = re.search(pattern, lowered_q)
            if not match:
                continue
            threshold = match.group(1)
            if threshold not in lowered_sql:
                issues.append("Threshold from question is missing in SQL")
                continue

            # More flexible checking - allow for variations
            if expected_op == ">=" and ">=" not in lowered_sql and ">" not in lowered_sql:
                issues.append("Comparator mismatch: expected >= or >")
            elif expected_op == "<=" and "<=" not in lowered_sql and "<" not in lowered_sql:
                issues.append("Comparator mismatch: expected <= or <")
            elif expected_op == ">" and re.search(r">=\s*" + re.escape(threshold), lowered_sql) is None and re.search(r">\s*" + re.escape(threshold), lowered_sql) is None:
                issues.append("Comparator mismatch: expected >")
            elif expected_op == "<" and re.search(r"<=\s*" + re.escape(threshold), lowered_sql) is None and re.search(r"<\s*" + re.escape(threshold), lowered_sql) is None:
                issues.append("Comparator mismatch: expected <")

        # Generic intent checks - but be less strict for education/company domain
        asks_count = re.search(r"\bcount\b|number of|how many", lowered_q) is not None
        # Allow exceptions for sales/quantity context
        sales_quantity_intent = re.search(r"\bsold\b|\bsales\b|\bselling\b|\bquantity\b", lowered_q) is not None
        sales_quantity_sql = (
            "sum(" in lowered_sql
            and any(token in lowered_sql for token in ["quantity", "invoice_item", "invoice"])
        )
        # Allow COUNT or aggregation for count requests
        has_aggregation = any(agg in lowered_sql for agg in ["count(", "sum(", "avg(", "max(", "min("])
        if asks_count and not has_aggregation and not sales_quantity_sql:
            if "count(" not in lowered_sql:
                issues.append("Question asks for a count but SQL does not use aggregation")

        # AVG check - less strict
        if re.search(r"\baverage\b|\bavg\b", lowered_q):
            if "avg(" not in lowered_sql and "average" not in lowered_sql and "sum(" not in lowered_sql:
                issues.append("Question asks for an average but SQL does not use AVG or SUM")

        # Average comparison - allow more flexibility
        asks_average_comparison = (
            re.search(r"\baverage\b|\bavg\b", lowered_q) is not None
            and re.search(r"greater than|more than|above|less than|below|higher than|lower than", lowered_q) is not None
        )
        asks_grouping = re.search(r"\beach\b|\bper\b|for each|in each|group by", lowered_q) is not None
        if asks_average_comparison and not asks_grouping:
            # Check if comparing against subquery average
            has_scalar_avg = ("(select avg(" in lowered_sql or "(select avg (" in lowered_sql or 
                            ("avg(" in lowered_sql and "group by" not in lowered_sql))
            if not has_scalar_avg and "avg(" not in lowered_sql:
                issues.append("Average comparison question should compare against a scalar AVG expression")

        # Ranking checks - be more flexible
        asks_ranking = re.search(r"\btop\b|\bhighest\b|\blowest\b|\brank\b|\bmost\b", lowered_q) is not None
        asks_partitioned_ranking = re.search(r"\beach\b|\bper\b|for each|in each|within each", lowered_q) is not None
        uses_window_ranking = (
            "over (" in lowered_sql
            and any(token in lowered_sql for token in ["row_number(", "rank(", "dense_rank("])
        )
        uses_max_or_min_extreme = "max(" in lowered_sql or "min(" in lowered_sql

        if asks_ranking:
            # Allow ORDER BY + LIMIT or MAX/MIN for ranking
            has_order_limit = "order by" in lowered_sql and "limit" in lowered_sql
            if not uses_window_ranking and not uses_max_or_min_extreme and not has_order_limit:
                issues.append("Ranking question should use ORDER BY and LIMIT or MAX/MIN")
            # Partitioned ranking - allow GROUP BY or PARTITION BY
            if asks_partitioned_ranking and not ("partition by" in lowered_sql or "group by" in lowered_sql):
                issues.append("Ranking in each group should use PARTITION BY or GROUP BY")

        # Grouping requirement - be less strict
        if re.search(r"\beach\b|\bper\b|for each|in each", lowered_q):
            latest_correlated_pattern = (
                "max(" in lowered_sql
                and "where" in lowered_sql
                and "_id" in lowered_sql
            )
            has_group_by = "group by" in lowered_sql
            if not has_group_by and not uses_window_ranking and not latest_correlated_pattern:
                # Check if it's a simple list query (not aggregation)
                if "count(" in lowered_sql or "sum(" in lowered_sql or "avg(" in lowered_sql:
                    issues.append("Grouping question should use GROUP BY")

        # Targeted guards for common company-domain intents - less strict
        asks_active_projects = re.search(r"active", lowered_q) is not None and re.search(r"project", lowered_q) is not None
        if asks_active_projects:
            if "active" not in lowered_sql:
                issues.append("Active projects question should filter projects by active status")

        # For simple listing questions, don't require all tables
        simple_list = re.search(r"list all|show all|retrieve all|display all|find all", lowered_q) is not None
        if not simple_list:
            asks_clients_projects = re.search(r"client", lowered_q) is not None and re.search(r"project", lowered_q) is not None
            if asks_clients_projects:
                if "client" not in lowered_sql or "project" not in lowered_sql:
                    issues.append("Clients and projects question should include both clients and projects tables")

        # Negation / Exclusion - important check
        if any(term in lowered_q for term in ["no ", "without", "does not have", "do not have", "have no"]):
            if not any(token in lowered_sql for token in ["left join", "not in", "not exists", "is null"]):
                issues.append("Negation question should use anti-join or exclusion logic")

        # Latest/Recent - should use ordering
        if re.search(r"latest|most recent|newest", lowered_q):
            if "order by" not in lowered_sql and "max(" not in lowered_sql:
                issues.append("Latest/most recent question should use ORDER BY DESC or MAX")

        # Simple domain sanity checks (not overly strict)
        return issues

    def _question_tokens(self, text: str) -> list[str]:
        tokens = [token for token in re.findall(r"[a-z_]+", text) if len(token) >= 3]
        return tokens

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z_]+", text.lower()) if len(token) >= 3}

    def _term_is_present(self, term: str, lowered_sql: str, sql_tokens: set[str]) -> bool:
        singular = self._singularize(term)
        plural = self._pluralize(singular)

        # Direct token/text presence.
        if term in sql_tokens or singular in sql_tokens or plural in sql_tokens:
            return True
        if term in lowered_sql or singular in lowered_sql or plural in lowered_sql:
            return True

        # Bridge-key presence (e.g., course_id satisfies "courses" in many schemas).
        fk_tokens = {f"{singular}_id", f"{plural}_id"}
        if any(token in sql_tokens for token in fk_tokens):
            return True

        return False

    def _singularize(self, term: str) -> str:
        if term.endswith("ies") and len(term) > 3:
            return term[:-3] + "y"
        if term.endswith("s") and not term.endswith("ss") and len(term) > 3:
            return term[:-1]
        return term

    def _pluralize(self, term: str) -> str:
        if term.endswith("y") and len(term) > 1:
            return term[:-1] + "ies"
        if term.endswith("s"):
            return term
        return term + "s"

    def _stopwords(self) -> set[str]:
        return {
            "the", "and", "with", "from", "that", "have", "has", "for", "each",
            "per", "who", "what", "when", "where", "which", "this", "those", "these",
            "all", "more", "than", "less", "most", "least", "top", "high", "highers",
            "highest", "lowest", "many", "number", "total", "average", "avg", "list",
            "show", "find", "get", "name", "names", "years", "year", "older", "younger",
            "student", "students", "employee", "employees", "customer", "customers", "department",
            "departments", "project", "projects", "client", "clients", "track", "tracks",
            "album", "albums", "artist", "artists", "genre", "genres", "invoice", "invoices",
        }