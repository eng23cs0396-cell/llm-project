from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FewShotExample:
    category: str
    question: str
    schema_hint: str
    sql: str
    keywords: list[str]


# Curated high-impact few-shot examples covering core SQL paradigms (Spider/BIRD aligned)
CURATED_FEW_SHOT_LIBRARY: list[FewShotExample] = [
    FewShotExample(
        category="basic_filter",
        question="Find all active students who enrolled after 2022.",
        schema_hint="students(id, name, is_active, enrollment_year, gpa)",
        sql="SELECT name, enrollment_year, gpa FROM students WHERE is_active = 1 AND enrollment_year > 2022;",
        keywords=["find", "all", "active", "enrolled", "after", "filter", "status"],
    ),
    FewShotExample(
        category="aggregation_group_by",
        question="How many courses are offered by each department? Show the department name and count.",
        schema_hint="departments(dept_id, dept_name), courses(course_id, course_name, dept_id)",
        sql="SELECT d.dept_name, COUNT(c.course_id) AS total_courses FROM departments d LEFT JOIN courses c ON d.dept_id = c.dept_id GROUP BY d.dept_id, d.dept_name ORDER BY total_courses DESC;",
        keywords=["how many", "count", "each", "per", "department", "by", "total"],
    ),
    FewShotExample(
        category="multi_join",
        question="What are the names of customers who placed orders for product 'Laptop' along with their order date?",
        schema_hint="customers(customer_id, customer_name), orders(order_id, customer_id, order_date), order_items(item_id, order_id, product_name)",
        sql="SELECT c.customer_name, o.order_date FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id WHERE oi.product_name = 'Laptop';",
        keywords=["along with", "with their", "and their", "orders", "customers", "product", "join"],
    ),
    FewShotExample(
        category="ranking_top_limit",
        question="Which publisher published the most titles in 2021? Give the publisher name.",
        schema_hint="publishers(pub_id, pub_name), titles(title_id, pub_id, pub_year)",
        sql="SELECT p.pub_name FROM publishers p JOIN titles t ON p.pub_id = t.pub_id WHERE t.pub_year = 2021 GROUP BY p.pub_id, p.pub_name ORDER BY COUNT(t.title_id) DESC LIMIT 1;",
        keywords=["most", "highest", "top", "rank", "maximum", "best", "limit"],
    ),
    FewShotExample(
        category="subquery_having",
        question="Find professors who advise more than 2 students and their average student GPA.",
        schema_hint="professors(prof_id, prof_name), student_advisors(student_id, prof_id), students(student_id, gpa)",
        sql="SELECT p.prof_name, AVG(s.gpa) AS avg_student_gpa FROM professors p JOIN student_advisors sa ON p.prof_id = sa.prof_id JOIN students s ON sa.student_id = s.student_id GROUP BY p.prof_id, p.prof_name HAVING COUNT(s.student_id) > 2;",
        keywords=["more than", "having", "average", "avg", "advise", "greater than", "threshold"],
    ),
    FewShotExample(
        category="temporal_date",
        question="List all sales transactions made in the year 2023 with total amount greater than 500.",
        schema_hint="sales(sale_id, customer_id, sale_date, amount)",
        sql="SELECT sale_id, customer_id, sale_date, amount FROM sales WHERE STRFTIME('%Y', sale_date) = '2023' AND amount > 500 ORDER BY amount DESC;",
        keywords=["year", "date", "temporal", "month", "strftime", "time", "recent", "sales"],
    ),
    FewShotExample(
        category="existence_subquery",
        question="Find all customers who have never placed an order.",
        schema_hint="customers(customer_id, customer_name, email), orders(order_id, customer_id, total_price)",
        sql="SELECT customer_name, email FROM customers WHERE customer_id NOT IN (SELECT DISTINCT customer_id FROM orders WHERE customer_id IS NOT NULL);",
        keywords=["never", "not in", "without", "no", "not have", "missing", "zero"],
    ),
]


class PromptEngine:
    """Dynamic prompt generation engine using schema grounding and dynamic few-shot selection."""

    def __init__(self, examples: list[FewShotExample] | None = None) -> None:
        self.examples = examples or CURATED_FEW_SHOT_LIBRARY

    def select_few_shot_examples(self, question: str, k: int = 3) -> list[FewShotExample]:
        """Select top-k most relevant few-shot examples based on lexical and intent scoring."""
        q_lower = question.lower()
        scored: list[tuple[float, FewShotExample]] = []

        for ex in self.examples:
            score = 0.0
            # Keyword matching
            for kw in ex.keywords:
                if kw in q_lower:
                    score += 2.0
                elif any(word in q_lower.split() for word in kw.split()):
                    score += 1.0

            # Intent checks
            if any(term in q_lower for term in ["how many", "count", "number of"]) and "count" in ex.sql.lower():
                score += 3.0
            if any(term in q_lower for term in ["top", "highest", "most", "lowest", "least"]) and "order by" in ex.sql.lower():
                score += 3.0
            if any(term in q_lower for term in ["per", "each", "group by"]) and "group by" in ex.sql.lower():
                score += 3.0
            if any(term in q_lower for term in ["year", "date", "month", "latest", "recent"]) and ("strftime" in ex.sql.lower() or "date" in ex.sql.lower()):
                score += 3.0
            if any(term in q_lower for term in ["never", "not", "without", "no "]) and ("not in" in ex.sql.lower() or "not exists" in ex.sql.lower()):
                score += 3.0

            scored.append((score, ex))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:k]]

    def format_schema_context(self, schema: dict[str, Any], dialect: str = "sqlite") -> str:
        """Format database schema into an organized, unambiguous context for the LLM."""
        tables = schema.get("tables", {})
        if not tables:
            return "No schema available."

        lines = [f"### Database Dialect: {dialect.upper()}", "### Schema:"]

        for table_name, table_info in tables.items():
            cols = table_info.get("columns", [])
            col_strs = []
            for col in cols:
                cname = col.get("name", "")
                ctype = col.get("type", "TEXT")
                pk = " PRIMARY KEY" if col.get("primary_key") else ""
                col_strs.append(f"  - {cname} ({ctype}{pk})")

            lines.append(f"Table `{table_name}`:")
            lines.extend(col_strs)

            fks = table_info.get("foreign_keys", [])
            if fks:
                lines.append("  Foreign Keys:")
                for fk in fks:
                    col_name = fk.get("column", "")
                    ref_tbl = fk.get("references_table", "")
                    ref_col = fk.get("references_column", "")
                    lines.append(f"    - `{col_name}` -> `{ref_tbl}`.`{ref_col}`")
            lines.append("")

        return "\n".join(lines).strip()

    def build_system_prompt(self, dialect: str = "sqlite") -> str:
        return (
            f"You are an expert Text-to-SQL engineer. Your task is to generate high-performance, "
            f"syntactically valid SQL queries tailored for {dialect.upper()}.\n\n"
            f"RULES:\n"
            f"1. Use ONLY the tables and columns explicitly present in the provided schema.\n"
            f"2. Pay close attention to Foreign Key relationships when joining tables.\n"
            f"3. Use table aliases (e.g. `t1`, `t2` or table initials) to avoid ambiguous column references.\n"
            f"4. For date/time comparisons in SQLite, use `STRFTIME()` or standard ISO format `YYYY-MM-DD`.\n"
            f"5. Return ONLY executable SQL. Do NOT add explanation, markdown formatting outside of ```sql ``` blocks, or comments.\n"
            f"6. Make sure strings in WHERE clauses are quoted accurately."
        )

    def build_user_prompt(
        self,
        question: str,
        schema_text: str,
        few_shot_examples: list[FewShotExample],
    ) -> str:
        sections = [f"Database Schema:\n{schema_text}\n"]

        if few_shot_examples:
            sections.append("### Relevant Examples:")
            for idx, ex in enumerate(few_shot_examples, 1):
                sections.append(
                    f"Example {idx}:\n"
                    f"Question: {ex.question}\n"
                    f"Schema: {ex.schema_hint}\n"
                    f"SQL:\n```sql\n{ex.sql}\n```\n"
                )

        sections.append(
            f"### Current Task:\n"
            f"Question: {question}\n"
            f"Generate the exact {schema_text.splitlines()[0] if schema_text else ''} query to answer this question.\n"
            f"SQL:"
        )

        return "\n".join(sections)

    def build_repair_prompt(
        self,
        question: str,
        failed_sql: str,
        error_message: str,
        schema_text: str,
        dialect: str = "sqlite",
    ) -> str:
        return (
            f"Database Schema:\n{schema_text}\n\n"
            f"The following SQL query was generated for the question but failed during execution on {dialect.upper()}:\n\n"
            f"Question: {question}\n"
            f"Failed SQL:\n```sql\n{failed_sql}\n```\n"
            f"Execution Error Message:\n{error_message}\n\n"
            f"Carefully analyze the error message (such as missing column, wrong table join, type error, or syntax mistake).\n"
            f"Provide the corrected, fully working SQL query inside a ```sql ``` block. Return ONLY the SQL query."
        )
