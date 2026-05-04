from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.retrieval_service import RetrievalContext
from app.storage import DatabaseMetadata


@dataclass
class QueryIntent:
    operation: str
    needs_join: bool = False
    needs_group_by: bool = False
    needs_aggregation: bool = False
    needs_ranking: bool = False
    needs_filter: bool = False
    comparator: str | None = None
    top_n: int | None = None
    needs_percentage: bool = False
    needs_window_rank: bool = False
    inferred_entities: list[str] = field(default_factory=list)
    inferred_metrics: list[str] = field(default_factory=list)


@dataclass
class SchemaRoleIndex:
    entity_tables: list[str]
    bridge_tables: list[str]
    metric_columns: dict[str, list[str]]
    label_columns: dict[str, list[str]]
    temporal_columns: dict[str, list[str]]
    status_columns: dict[str, list[str]]
    identifier_columns: dict[str, list[str]]


class SQLGenerator:
    def generate(self, question: str, metadata: DatabaseMetadata, context: RetrievalContext) -> str:
        lowered = question.lower()

        intent = self._classify_intent(lowered, metadata, context)
        role_index = self._infer_schema_roles(metadata)

        mentioned_tables = self._mentioned_tables(lowered, context.relevant_tables)
        primary_table = self._primary_table_from_question(lowered, metadata, context)

        # Prefer direct single-table filtering when only one entity is requested.
        if primary_table:
            primary_where = self._try_generate_where_clause(lowered, metadata, primary_table)
            name_only_request = (
                self._contains_any(lowered, ["names of", "name of"])
                and self._contains_any(lowered, ["all", "list", "show", "display", "retrieve"])
            )
            if (
                primary_where
                and not intent.needs_group_by
                and not intent.needs_ranking
                and not self._contains_any(lowered, ["along with", "with their", "for each", "in each"]) 
                and not self._contains_any(lowered, ["enrolled in more than", "teach more than", "teaches more than"])
            ):
                if name_only_request:
                    name_col = self._pick_name_like_column(metadata, primary_table)
                    select_cols = name_col if name_col else "*"
                elif self._contains_any(lowered, ["all", "list all", "show all", "retrieve all", "display all"]):
                    select_cols = "*"
                else:
                    select_cols = "*"
                return f"SELECT {select_cols} FROM {primary_table} WHERE {primary_where}"

        planned_sql = self._try_generate_schema_graph_plan_sql(
            lowered,
            metadata,
            context,
            intent,
            role_index,
        )
        if planned_sql:
            return planned_sql

        lookup_sql = self._try_generate_lookup_by_id_sql(lowered, metadata, context)
        if lookup_sql:
            return lookup_sql

        relation_projection_sql = self._try_generate_relation_projection_sql(lowered, metadata, context)
        if relation_projection_sql:
            return relation_projection_sql

        max_metric_sql = self._try_generate_max_metric_sql(lowered, metadata, context)
        if max_metric_sql:
            return max_metric_sql

        latest_per_entity_sql = self._try_generate_latest_per_entity_sql(lowered, metadata, context)
        if latest_per_entity_sql:
            return latest_per_entity_sql

        absent_on_date_sql = self._try_generate_absent_on_date_sql(lowered, metadata, context)
        if absent_on_date_sql:
            return absent_on_date_sql

        manager_team_size_sql = self._try_generate_manager_team_size_sql(lowered, metadata, context)
        if manager_team_size_sql:
            return manager_team_size_sql

        avg_metric_by_group_sql = self._try_generate_avg_metric_by_group_sql(lowered, metadata)
        if avg_metric_by_group_sql:
            return avg_metric_by_group_sql

        total_hours_sql = self._try_generate_total_hours_by_entity_sql(lowered, metadata)
        if total_hours_sql:
            return total_hours_sql

        above_avg_sql = self._try_generate_above_average_sql(lowered, metadata, context)
        if above_avg_sql:
            return above_avg_sql

        at_least_one_sql = self._try_generate_at_least_one_sql(lowered, metadata, context)
        if at_least_one_sql:
            return at_least_one_sql

        percentage_sql = self._try_generate_percentage_sql(
            lowered,
            metadata,
            context,
            intent=intent,
            role_index=role_index,
        )
        if percentage_sql:
            return percentage_sql

        analytic_sql = self._try_generate_analytic_sql(
            lowered,
            metadata,
            context,
            intent=intent,
            role_index=role_index,
        )
        if analytic_sql:
            return analytic_sql

        # Check for GROUP BY patterns (aggregation queries)
        groupby_sql = self._try_generate_groupby_sql(
            lowered,
            metadata,
            context,
            intent=intent,
            role_index=role_index,
        )
        if groupby_sql:
            return groupby_sql

        # Check for JOIN patterns
        join_sql = self._try_generate_join_sql(
            lowered,
            metadata,
            context,
            intent=intent,
            role_index=role_index,
        )
        if join_sql:
            return join_sql

        # Check for WHERE patterns (filtering)
        table = self._choose_best_table(
            lowered,
            metadata,
            context,
            intent=intent,
            role_index=role_index,
        )
        where_clause = self._try_generate_where_clause(lowered, metadata, table)

        # Simple aggregation without GROUP BY
        if intent.operation == "count" or "how many" in lowered:
            if where_clause:
                return f"SELECT COUNT(*) AS row_count FROM {table} WHERE {where_clause}"
            return f"SELECT COUNT(*) AS row_count FROM {table}"

        if intent.operation == "average" or "avg" in lowered:
            metric_col = self._pick_numeric_column(metadata, table, lowered) or "*"
            if where_clause:
                return f"SELECT AVG({metric_col}) AS average_value FROM {table} WHERE {where_clause}"
            return f"SELECT AVG({metric_col}) AS average_value FROM {table}"

        # Check for ORDER BY patterns (top, highest, etc.)
        order_sql = self._try_generate_order_by_sql(
            lowered,
            metadata,
            context,
            table,
            intent=intent,
            role_index=role_index,
        )
        if order_sql:
            return order_sql

        # Keep simple projection late so it does not eclipse richer join/aggregation/ranking patterns.
        simple_sql = self._try_generate_simple_select_sql(lowered, metadata, context)
        if simple_sql:
            return simple_sql

        columns = self._choose_projection_columns(lowered, metadata, table, context)
        select_cols = ", ".join(columns[:5]) if columns else "*"
        
        # Check for DISTINCT
        if intent.operation == "distinct" or ("show" in lowered and "names" in lowered):
            select_clause = f"DISTINCT {', '.join(columns)}" if columns else "DISTINCT *"
            if where_clause:
                return f"SELECT {select_clause} FROM {table} WHERE {where_clause}"
            return f"SELECT {select_clause} FROM {table}"
        
        if table:
            if where_clause:
                return f"SELECT {select_cols} FROM {table} WHERE {where_clause}"
            return f"SELECT {select_cols} FROM {table}"
        return "SELECT 1"

    def _try_generate_relation_projection_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        if not self._contains_any(lowered_question, ["with their", "along with", "together with", "and their"]):
            return None

        # Count/aggregation requests should flow to GROUP BY logic, not plain relation projection.
        if self._contains_any(lowered_question, ["count", "number", "how many", "total", "avg", "average", "sum"]):
            return None

        primary_table = self._primary_table_from_question(lowered_question, metadata, context)
        mentioned = self._mentioned_tables(lowered_question, list(metadata.schema.get("tables", {}).keys()))
        inferred_related = None
        if len(mentioned) < 2 and primary_table:
            inferred_related = self._infer_related_table_from_primary(lowered_question, metadata, primary_table)
            if inferred_related:
                mentioned = [primary_table, inferred_related]

        if primary_table and primary_table in mentioned:
            mentioned = [primary_table] + [table for table in mentioned if table != primary_table]

        if len(mentioned) < 2:
            return None

        left_table, right_table = mentioned[0], mentioned[1]
        detail_phrase = self._extract_relation_detail_phrase(lowered_question)
        left_label = self._pick_name_like_column(metadata, left_table) or self._find_table_id_column(metadata, left_table)
        right_label = self._pick_name_like_column(metadata, right_table)
        if not right_label:
            right_label = self._pick_question_aligned_column(
                lowered_question,
                metadata,
                right_table,
                prefer_temporal=False,
                avoid_identifier=True,
            )
        if not right_label:
            right_label = self._pick_question_aligned_column(
                lowered_question,
                metadata,
                right_table,
                prefer_temporal=True,
                avoid_identifier=True,
            )
        if not right_label:
            right_label = self._pick_temporal_like_column(metadata, right_table)
        if not right_label:
            right_label = self._find_table_id_column(metadata, right_table)
        left_id = self._find_table_id_column(metadata, left_table)
        right_id = self._find_table_id_column(metadata, right_table)
        if not left_label or not right_label or not left_id or not right_id:
            return None

        phrase_detail_col = self._pick_table_column_from_phrase(metadata, left_table, detail_phrase)
        if phrase_detail_col and phrase_detail_col != left_label:
            return f"SELECT l.{left_label}, l.{phrase_detail_col} FROM {left_table} l"

        # If requested attributes are all available on the left entity, avoid an unnecessary join.
        # UNLESS the right_table is explicitly mentioned (e.g., "students along with courses").
        requested_detail_col = self._pick_question_aligned_column(
            lowered_question,
            metadata,
            left_table,
            prefer_temporal=True,
            avoid_identifier=True,
        )
        right_table_explicitly_mentioned = self._table_mentioned_in_question(lowered_question, right_table) or bool(
            inferred_related and right_table == inferred_related
        )
        if (
            requested_detail_col 
            and requested_detail_col != left_label
            and not right_table_explicitly_mentioned
        ):
            return f"SELECT l.{left_label}, l.{requested_detail_col} FROM {left_table} l"

        direct_lr = self._find_fk_to_table(metadata, left_table, right_table)
        if direct_lr:
            return (
                f"SELECT l.{left_label}, r.{right_label} "
                f"FROM {left_table} l JOIN {right_table} r ON l.{direct_lr} = r.{right_id}"
            )

        direct_rl = self._find_fk_to_table(metadata, right_table, left_table)
        if direct_rl:
            return (
                f"SELECT l.{left_label}, r.{right_label} "
                f"FROM {left_table} l JOIN {right_table} r ON r.{direct_rl} = l.{left_id}"
            )

        bridge = self._find_bridge_table(metadata, left_table, right_table)
        if bridge:
            bridge_table, fk_left, fk_right = bridge
            return (
                f"SELECT l.{left_label}, r.{right_label} "
                f"FROM {left_table} l "
                f"JOIN {bridge_table} b ON b.{fk_left} = l.{left_id} "
                f"JOIN {right_table} r ON b.{fk_right} = r.{right_id}"
            )

        return None

    def _try_generate_max_metric_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        if not self._contains_any(lowered_question, ["highest", "top", "maximum", "max"]):
            return None

        if self._contains_any(lowered_question, ["average", "avg", "mean"]):
            return None

        if "second" in lowered_question and "highest" in lowered_question:
            return None

        if self._contains_any(lowered_question, ["each", "per", "in each", "within each"]):
            return None

        if not self._contains_any(lowered_question, ["rating", "score", "grade", "salary", "amount"]):
            return None

        candidate_tables = context.relevant_tables or list(metadata.schema.get("tables", {}).keys())
        best: tuple[str, str] | None = None
        for table in candidate_tables:
            cols = [str(col.get("name", "")) for col in metadata.schema.get("tables", {}).get(table, {}).get("columns", [])]
            metric_col = None
            for col in cols:
                col_lower = col.lower()
                if any(token in col_lower for token in ["rating", "score", "grade", "salary", "amount"]):
                    metric_col = col
                    break
            if metric_col:
                best = (table, metric_col)
                break

        if not best:
            return None

        table, metric_col = best
        id_col = self._find_table_id_column(metadata, table)
        label_col = self._pick_name_like_column(metadata, table)
        selected = label_col or id_col or metric_col
        if not selected:
            return None

        return (
            f"SELECT {selected}, {metric_col} "
            f"FROM {table} "
            f"WHERE {metric_col} = (SELECT MAX({metric_col}) FROM {table})"
        )

    def _try_generate_latest_per_entity_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        if not self._contains_any(lowered_question, ["latest", "most recent", "newest"]):
            return None
        if not self._contains_any(lowered_question, ["each", "per", "for each"]):
            return None

        tables = metadata.schema.get("tables", {})
        for table_name, info in tables.items():
            cols = [str(col.get("name", "")) for col in info.get("columns", [])]
            date_col = next((c for c in cols if any(k in c.lower() for k in ["date", "time", "created", "updated", "effective"])), None)
            if not date_col:
                continue

            metric_col = next((c for c in cols if any(k in c.lower() for k in ["salary", "amount", "value", "score", "rating"])), None)
            if not metric_col:
                continue

            entity_fk = next((c for c in cols if c.lower().endswith("_id") and c.lower() != self._find_table_id_column(metadata, table_name)), None)
            if not entity_fk:
                continue

            return (
                f"SELECT {entity_fk}, {metric_col} "
                f"FROM {table_name} t1 "
                f"WHERE {date_col} = ("
                f"SELECT MAX({date_col}) FROM {table_name} t2 WHERE t1.{entity_fk} = t2.{entity_fk}"
                ")"
            )

        return None

    def _try_generate_absent_on_date_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        if "absent" not in lowered_question:
            return None

        date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", lowered_question)
        date_literal = date_match.group(0) if date_match else None
        if not date_literal:
            return None

        for table, info in metadata.schema.get("tables", {}).items():
            cols = [str(col.get("name", "")) for col in info.get("columns", [])]
            status_col = next((c for c in cols if "status" in c.lower()), None)
            date_col = next((c for c in cols if c.lower() == "date" or "date" in c.lower()), None)
            emp_col = next((c for c in cols if c.lower().endswith("_id")), None)
            if status_col and date_col and emp_col:
                return (
                    f"SELECT {emp_col} FROM {table} "
                    f"WHERE {date_col} = '{date_literal}' AND {status_col} = 'Absent'"
                )
        return None

    def _try_generate_manager_team_size_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        if "manager" not in lowered_question or "team" not in lowered_question:
            return None

        for table, info in metadata.schema.get("tables", {}).items():
            cols = [str(col.get("name", "")) for col in info.get("columns", [])]
            manager_col = next((c for c in cols if "manager" in c.lower() and c.lower().endswith("_id")), None)
            if manager_col:
                return (
                    f"SELECT {manager_col}, COUNT(*) AS team_size "
                    f"FROM {table} "
                    f"WHERE {manager_col} IS NOT NULL "
                    f"GROUP BY {manager_col}"
                )
        return None

    def _try_generate_schema_graph_plan_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
        intent: QueryIntent,
        role_index: SchemaRoleIndex,
    ) -> str | None:
        primary_table = self._primary_table_from_question(lowered_question, metadata, context)
        ordered_tables = self._ordered_table_mentions(lowered_question, metadata)
        if primary_table and primary_table in ordered_tables:
            ordered_tables = [primary_table] + [table for table in ordered_tables if table != primary_table]
        if len(ordered_tables) < 2 and primary_table:
            inferred_related = self._infer_related_table_from_primary(lowered_question, metadata, primary_table)
            if inferred_related:
                ordered_tables = [primary_table, inferred_related]
        relationships = self._infer_all_relationships(metadata)

        # Relationship projection for "X with their Y" style prompts.
        if self._contains_any(lowered_question, ["with their", "along with", "together with", "and their"]) and len(ordered_tables) >= 2:
            if self._contains_any(lowered_question, ["count", "number", "how many", "total", "avg", "average", "sum"]):
                return None
            left_table, right_table = ordered_tables[0], ordered_tables[1]
            detail_phrase = self._extract_relation_detail_phrase(lowered_question)
            left_id = self._find_table_id_column(metadata, left_table)
            right_id = self._find_table_id_column(metadata, right_table)
            left_label = self._pick_name_like_column(metadata, left_table) or left_id
            right_label = self._pick_name_like_column(metadata, right_table)
            if not right_label:
                right_label = self._pick_question_aligned_column(
                    lowered_question,
                    metadata,
                    right_table,
                    prefer_temporal=False,
                    avoid_identifier=True,
                )
            if not right_label:
                right_label = self._pick_question_aligned_column(
                    lowered_question,
                    metadata,
                    right_table,
                    prefer_temporal=True,
                    avoid_identifier=True,
                )
            if not right_label:
                right_label = self._pick_temporal_like_column(metadata, right_table)
            if not right_label:
                right_label = right_id
            if left_id and right_id and left_label and right_label:
                phrase_detail_col = self._pick_table_column_from_phrase(metadata, left_table, detail_phrase)
                if phrase_detail_col and phrase_detail_col != left_label:
                    return f"SELECT l.{left_label}, l.{phrase_detail_col} FROM {left_table} l"

                requested_detail_col = self._pick_question_aligned_column(
                    lowered_question,
                    metadata,
                    left_table,
                    prefer_temporal=True,
                    avoid_identifier=True,
                )
                # Don't skip the join if the right_table is explicitly mentioned
                right_table_explicitly_mentioned = self._table_mentioned_in_question(lowered_question, right_table) or bool(
                    inferred_related and right_table == inferred_related
                )
                if (
                    requested_detail_col 
                    and requested_detail_col != left_label
                    and not right_table_explicitly_mentioned
                ):
                    return f"SELECT l.{left_label}, l.{requested_detail_col} FROM {left_table} l"

                if (left_table, right_table) in relationships:
                    fk_col, target_col = relationships[(left_table, right_table)]
                    return (
                        f"SELECT l.{left_label}, r.{right_label} "
                        f"FROM {left_table} l JOIN {right_table} r ON l.{fk_col} = r.{target_col}"
                    )
                if (right_table, left_table) in relationships:
                    fk_col, target_col = relationships[(right_table, left_table)]
                    return (
                        f"SELECT l.{left_label}, r.{right_label} "
                        f"FROM {left_table} l JOIN {right_table} r ON r.{fk_col} = l.{target_col}"
                    )

                bridge = self._find_bridge_table(metadata, left_table, right_table)
                if bridge:
                    bridge_table, fk_left, fk_right = bridge
                    return (
                        f"SELECT l.{left_label}, r.{right_label} "
                        f"FROM {left_table} l "
                        f"JOIN {bridge_table} b ON b.{fk_left} = l.{left_id} "
                        f"JOIN {right_table} r ON b.{fk_right} = r.{right_id}"
                    )

        # Average metric by group via graph: AVG(metric_table.metric) grouped by group_table.label.
        if self._contains_any(lowered_question, ["average", "avg"]) and self._contains_any(lowered_question, ["each", "per", "by"]):
            metric_table, metric_col = self._find_best_metric_column_for_question(lowered_question, metadata, role_index)
            if metric_table and metric_col:
                group_table = self._find_group_table_for_question(lowered_question, metadata, role_index)
                if not group_table and ordered_tables:
                    group_table = ordered_tables[0]
                if group_table:
                    group_id = self._find_table_id_column(metadata, group_table)
                    group_label = self._pick_name_like_column(metadata, group_table) or group_id
                    if group_id and group_label:
                        select_group_col = f"g.{group_label}"
                        group_by_cols = [f"g.{group_id}"]
                        if group_label != group_id:
                            group_by_cols.append(select_group_col)
                        # direct metric->group
                        fk_direct = self._find_fk_to_table(metadata, metric_table, group_table)
                        if fk_direct:
                            return (
                                f"SELECT {select_group_col}, AVG(m.{metric_col}) AS average_grade "
                                f"FROM {group_table} g JOIN {metric_table} m ON m.{fk_direct} = g.{group_id} "
                                f"GROUP BY g.{group_id}"
                            )

                        # metric->entity->group
                        for entity_table in role_index.entity_tables:
                            fk_metric_entity = self._find_fk_to_table(metadata, metric_table, entity_table)
                            fk_entity_group = self._find_fk_to_table(metadata, entity_table, group_table)
                            entity_id = self._find_table_id_column(metadata, entity_table)
                            if fk_metric_entity and fk_entity_group and entity_id:
                                return (
                                    f"SELECT {select_group_col}, AVG(m.{metric_col}) AS average_grade "
                                    f"FROM {group_table} g "
                                    f"JOIN {entity_table} e ON e.{fk_entity_group} = g.{group_id} "
                                    f"JOIN {metric_table} m ON m.{fk_metric_entity} = e.{entity_id} "
                                    f"GROUP BY g.{group_id}"
                                )

        # Highest/top average metric by group (e.g., course with highest average grade).
        if self._contains_any(lowered_question, ["average", "avg"]) and self._contains_any(lowered_question, ["highest", "top", "maximum", "max"]):
            metric_table, metric_col = self._find_best_metric_column_for_question(lowered_question, metadata, role_index)
            if metric_table and metric_col:
                group_table = self._find_group_table_for_question(lowered_question, metadata, role_index)
                if (not group_table or group_table == metric_table) and ordered_tables:
                    for t in ordered_tables:
                        if t != metric_table:
                            group_table = t
                            break
                if group_table and group_table != metric_table:
                    group_id = self._find_table_id_column(metadata, group_table)
                    group_label = self._pick_name_like_column(metadata, group_table) or group_id
                    if group_id and group_label:
                        select_group_col = f"g.{group_label}"
                        group_by_cols = [f"g.{group_id}"]
                        if group_label != group_id:
                            group_by_cols.append(select_group_col)

                        fk_direct = self._find_fk_to_table(metadata, metric_table, group_table)
                        if fk_direct:
                            top_n = self._extract_top_n(lowered_question, default_value=1)
                            if top_n == 1:
                                return (
                                    f"SELECT {select_group_col} "
                                    f"FROM {group_table} g JOIN {metric_table} m ON m.{fk_direct} = g.{group_id} "
                                    f"GROUP BY g.{group_id} ORDER BY AVG(m.{metric_col}) DESC LIMIT 1"
                                )
                            return (
                                f"SELECT {select_group_col}, AVG(m.{metric_col}) AS average_value "
                                f"FROM {group_table} g JOIN {metric_table} m ON m.{fk_direct} = g.{group_id} "
                                f"GROUP BY g.{group_id}, {select_group_col} ORDER BY average_value DESC LIMIT {top_n}"
                            )

                        for entity_table in role_index.entity_tables:
                            fk_metric_entity = self._find_fk_to_table(metadata, metric_table, entity_table)
                            fk_entity_group = self._find_fk_to_table(metadata, entity_table, group_table)
                            entity_id = self._find_table_id_column(metadata, entity_table)
                            if fk_metric_entity and fk_entity_group and entity_id:
                                top_n = self._extract_top_n(lowered_question, default_value=1)
                                if top_n == 1:
                                    return (
                                        f"SELECT {select_group_col} "
                                        f"FROM {group_table} g "
                                        f"JOIN {entity_table} e ON e.{fk_entity_group} = g.{group_id} "
                                        f"JOIN {metric_table} m ON m.{fk_metric_entity} = e.{entity_id} "
                                        f"GROUP BY g.{group_id} ORDER BY AVG(m.{metric_col}) DESC LIMIT 1"
                                    )
                                return (
                                    f"SELECT {select_group_col}, AVG(m.{metric_col}) AS average_value "
                                    f"FROM {group_table} g "
                                    f"JOIN {entity_table} e ON e.{fk_entity_group} = g.{group_id} "
                                    f"JOIN {metric_table} m ON m.{fk_metric_entity} = e.{entity_id} "
                                    f"GROUP BY g.{group_id}, {select_group_col} ORDER BY average_value DESC LIMIT {top_n}"
                                )

        # Latest metric per entity via correlated MAX(date).
        if self._contains_any(lowered_question, ["latest", "most recent", "newest"]) and self._contains_any(lowered_question, ["each", "per", "for each"]):
            for table_name, cols in role_index.temporal_columns.items():
                if not cols:
                    continue
                entity_fk = self._find_fk_to_any_entity(metadata, table_name, role_index)
                if not entity_fk:
                    continue
                metric_col = self._best_metric_for_table_question(lowered_question, metadata, table_name, role_index)
                if not metric_col:
                    continue
                date_col = cols[0]
                return (
                    f"SELECT {entity_fk}, {metric_col} "
                    f"FROM {table_name} t1 "
                    f"WHERE {date_col} = (SELECT MAX({date_col}) FROM {table_name} t2 WHERE t1.{entity_fk} = t2.{entity_fk})"
                )

        # Highest/top metric using scalar MAX when metric is explicit.
        if self._contains_any(lowered_question, ["highest", "top", "maximum", "max"]) and not self._contains_any(lowered_question, ["average", "avg", "mean"]):
            if "second" in lowered_question or self._contains_any(lowered_question, ["each", "per", "in each", "within each"]):
                return None
            metric_table, metric_col = self._find_best_metric_column_for_question(lowered_question, metadata, role_index)
            if metric_table and metric_col:
                id_col = self._find_table_id_column(metadata, metric_table)
                label_col = self._pick_name_like_column(metadata, metric_table)
                projection = label_col or id_col or metric_col
                if projection:
                    return (
                        f"SELECT {projection}, {metric_col} "
                        f"FROM {metric_table} "
                        f"WHERE {metric_col} = (SELECT MAX({metric_col}) FROM {metric_table})"
                    )

        return None

    def _ordered_table_mentions(self, lowered_question: str, metadata: DatabaseMetadata) -> list[str]:
        matches: list[tuple[int, str]] = []
        for table in metadata.schema.get("tables", {}).keys():
            t = table.lower()
            singular = t[:-1] if t.endswith("s") else t
            pos = lowered_question.find(t)
            if pos < 0:
                pos = lowered_question.find(singular)
            if pos >= 0:
                matches.append((pos, table))
        matches.sort(key=lambda x: x[0])
        return [table for _, table in matches]

    def _find_best_metric_column_for_question(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        role_index: SchemaRoleIndex,
    ) -> tuple[str | None, str | None]:
        best_score = 0.0
        best: tuple[str | None, str | None] = (None, None)
        for table_name, metric_cols in role_index.metric_columns.items():
            for col in metric_cols:
                score = self._score_column_relevance(lowered_question, col)
                col_lower = col.lower()
                if col_lower in lowered_question:
                    score += 3.0
                if any(token in lowered_question for token in ["salary", "rating", "score", "grade", "amount", "hours"]):
                    if any(token in col_lower for token in ["salary", "rating", "score", "grade", "amount", "hours"]):
                        score += 2.0
                if score > best_score:
                    best_score = score
                    best = (table_name, col)
        return best

    def _find_group_table_for_question(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        role_index: SchemaRoleIndex,
    ) -> str | None:
        tokens = self._tokens(lowered_question)
        best_table = None
        best_score = 0
        for table_name in metadata.schema.get("tables", {}).keys():
            table_tokens = self._tokens(table_name)
            score = len(tokens & table_tokens)
            for label_col in role_index.label_columns.get(table_name, []):
                score += len(tokens & self._tokens(label_col))
            if score > best_score:
                best_score = score
                best_table = table_name
        return best_table

    def _find_fk_to_any_entity(
        self,
        metadata: DatabaseMetadata,
        source_table: str,
        role_index: SchemaRoleIndex,
    ) -> str | None:
        for entity_table in role_index.entity_tables:
            fk = self._find_fk_to_table(metadata, source_table, entity_table)
            if fk:
                return fk
        return None

    def _infer_related_table_from_primary(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        primary_table: str,
    ) -> str | None:
        table_info = metadata.schema.get("tables", {}).get(primary_table, {})
        question_tokens = self._tokens(lowered_question)
        best_score = 0
        best_table = None

        for fk in table_info.get("foreign_keys", []):
            target_table = str(fk.get("references_table", ""))
            fk_col = str(fk.get("column", ""))
            if not target_table or not fk_col:
                continue

            score = 0
            if self._table_mentioned_in_question(lowered_question, target_table):
                score += 4

            fk_tokens: set[str] = set()
            for piece in re.split(r"[_\s]+", fk_col.lower()):
                if not piece or piece == "id":
                    continue
                fk_tokens.add(piece)
                fk_tokens.add(f"{piece}s")
                if piece.endswith("id") and len(piece) > 2:
                    fk_tokens.add(piece[:-2])
                if piece.endswith("s") and len(piece) > 1:
                    fk_tokens.add(piece[:-1])
            target_tokens = self._tokens(target_table)
            score += len(question_tokens & fk_tokens) * 3
            score += len(question_tokens & target_tokens) * 2

            if score > best_score:
                best_score = score
                best_table = target_table

        return best_table

    def _best_metric_for_table_question(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        table_name: str,
        role_index: SchemaRoleIndex,
    ) -> str | None:
        metric_cols = role_index.metric_columns.get(table_name, [])
        if not metric_cols:
            return None
        scored = sorted(
            ((self._score_column_relevance(lowered_question, col), col) for col in metric_cols),
            reverse=True,
        )
        return scored[0][1] if scored else metric_cols[0]

    def _classify_intent(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> QueryIntent:
        operation = "select"
        needs_percentage = self._contains_any(lowered_question, ["percentage", "percent", "ratio", "out of"])
        if self._contains_any(lowered_question, ["count", "how many", "number of"]):
            operation = "count"
        elif self._contains_any(lowered_question, ["average", "avg", "mean"]):
            operation = "average"
        elif self._contains_any(lowered_question, ["sum", "total", "quantity", "overall"]):
            operation = "sum"
        elif self._contains_any(lowered_question, ["maximum", "minimum", "max", "min", "highest", "lowest"]):
            operation = "extreme"
        elif self._contains_any(lowered_question, ["distinct", "unique"]):
            operation = "distinct"
        elif needs_percentage:
            operation = "percentage"

        needs_join = self._contains_any(
            lowered_question,
            ["along with", "with their", "together with", "including", "assigned", "enrolled"],
        ) or len(context.relevant_tables) > 1
        needs_group_by = self._contains_any(
            lowered_question,
            ["in each", "for each", "per", "group by", "each"],
        )
        needs_aggregation = operation in {"count", "average", "sum", "extreme", "percentage"}
        needs_ranking = self._contains_any(
            lowered_question,
            ["top", "highest", "lowest", "most", "least", "oldest", "youngest", "rank", "second"],
        )
        needs_window_rank = self._contains_any(
            lowered_question,
            ["within each", "in each", "per", "rank", "top-performing", "second highest"],
        )
        needs_filter = self._contains_any(
            lowered_question,
            ["where", "with", "after", "before", "greater than", "less than", "more than", "from", "in"],
        )

        comparator = None
        if self._contains_any(lowered_question, ["greater than", "more than", "above", "higher"]):
            comparator = ">"
        elif self._contains_any(lowered_question, ["less than", "below", "lower", "under"]):
            comparator = "<"
        elif self._contains_any(lowered_question, ["equal", "equals", "exactly"]):
            comparator = "="

        inferred_entities: list[str] = []
        inferred_metrics: list[str] = []
        question_tokens = self._tokens(lowered_question)

        table_names = context.relevant_tables or list(metadata.schema.get("tables", {}).keys())
        for table in table_names:
            table_tokens = self._tokens(table)
            if question_tokens & table_tokens:
                inferred_entities.append(table)

            for col in metadata.schema.get("tables", {}).get(table, {}).get("columns", []):
                col_name = str(col.get("name", ""))
                col_type = str(col.get("type", "")).lower()
                if not any(token in col_type for token in ["int", "real", "num", "decimal", "float", "double"]):
                    continue
                if "id" in col_name.lower():
                    continue
                if question_tokens & self._tokens(col_name):
                    inferred_metrics.append(col_name.lower())

        return QueryIntent(
            operation=operation,
            needs_join=needs_join,
            needs_group_by=needs_group_by,
            needs_aggregation=needs_aggregation,
            needs_ranking=needs_ranking,
            needs_filter=needs_filter,
            comparator=comparator,
            top_n=self._extract_top_n(lowered_question, default_value=10) if needs_ranking else None,
            needs_percentage=needs_percentage,
            needs_window_rank=needs_window_rank,
            inferred_entities=inferred_entities,
            inferred_metrics=list(dict.fromkeys(inferred_metrics)),
        )

    def _infer_schema_roles(self, metadata: DatabaseMetadata) -> SchemaRoleIndex:
        entity_tables: list[str] = []
        bridge_tables: list[str] = []
        metric_columns: dict[str, list[str]] = {}
        label_columns: dict[str, list[str]] = {}
        temporal_columns: dict[str, list[str]] = {}
        status_columns: dict[str, list[str]] = {}
        identifier_columns: dict[str, list[str]] = {}

        table_map = metadata.schema.get("tables", {})
        for table_name, table_info in table_map.items():
            columns = table_info.get("columns", [])
            fks = table_info.get("foreign_keys", [])

            fk_like_count = 0
            labels: list[str] = []
            metrics: list[str] = []
            temporals: list[str] = []
            statuses: list[str] = []
            ids: list[str] = []

            for col in columns:
                col_name = str(col.get("name", ""))
                col_lower = col_name.lower()
                col_type = str(col.get("type", "")).lower()

                if col_lower.endswith("_id") or col_lower == "id":
                    fk_like_count += 1
                    ids.append(col_name)

                if any(token in col_lower for token in ["name", "title", "label", "description"]):
                    labels.append(col_name)

                if any(token in col_type for token in ["int", "real", "num", "decimal", "float", "double"]):
                    if "id" not in col_lower:
                        metrics.append(col_name)

                if any(token in col_lower for token in ["date", "time", "created", "updated", "timestamp", "year"]):
                    temporals.append(col_name)

                if any(token in col_lower for token in ["status", "state", "active", "enabled", "flag"]):
                    statuses.append(col_name)

            if len(fks) >= 2 or (len(fks) >= 1 and fk_like_count >= 2 and not labels):
                bridge_tables.append(table_name)
            else:
                entity_tables.append(table_name)

            metric_columns[table_name] = metrics
            label_columns[table_name] = labels
            temporal_columns[table_name] = temporals
            status_columns[table_name] = statuses
            identifier_columns[table_name] = ids

        return SchemaRoleIndex(
            entity_tables=entity_tables,
            bridge_tables=bridge_tables,
            metric_columns=metric_columns,
            label_columns=label_columns,
            temporal_columns=temporal_columns,
            status_columns=status_columns,
            identifier_columns=identifier_columns,
        )

    def _try_generate_intent_sql(self, lowered_question: str, metadata: DatabaseMetadata) -> str | None:
        # DEPRECATED: Hardcoded database-specific logic. Use generic methods instead.
        return None

    def _try_generate_percentage_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
        intent: QueryIntent | None = None,
        role_index: SchemaRoleIndex | None = None,
    ) -> str | None:
        role_index = role_index or self._infer_schema_roles(metadata)
        if not (intent and intent.needs_percentage):
            return None

        table = self._choose_best_table(
            lowered_question,
            metadata,
            context,
            prioritize_metric_table=True,
            intent=intent,
            role_index=role_index,
        )
        if not table:
            return None

        metrics = list(role_index.metric_columns.get(table, []))
        if len(metrics) < 2:
            numeric = self._pick_numeric_column(metadata, table, lowered_question)
            if numeric:
                metrics = [numeric] + [m for m in metrics if m != numeric]
        if len(metrics) < 2:
            return None

        scored_metrics: list[tuple[float, str]] = []
        for col in metrics:
            score = self._score_column_relevance(lowered_question, col.lower())
            if any(token in col.lower() for token in ["obtained", "earned", "actual", "score", "marks", "grade"]):
                score += 2.0
            if any(token in col.lower() for token in ["total", "max", "possible", "full"]):
                score += 1.5
            scored_metrics.append((score, col))
        scored_metrics.sort(reverse=True)

        numerator = scored_metrics[0][1]
        denominator = None
        for _, candidate in scored_metrics[1:]:
            if candidate != numerator:
                denominator = candidate
                break
        if not denominator:
            return None

        label = self._pick_name_like_column(metadata, table)
        expr = f"ROUND(({numerator} * 100.0) / NULLIF({denominator}, 0), 2)"
        if label:
            return f"SELECT {label}, {expr} AS percentage_value FROM {table}"
        return f"SELECT {expr} AS percentage_value FROM {table}"

    def _try_generate_analytic_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
        intent: QueryIntent | None = None,
        role_index: SchemaRoleIndex | None = None,
    ) -> str | None:
        role_index = role_index or self._infer_schema_roles(metadata)
        if not (intent and (intent.needs_window_rank or intent.needs_ranking)):
            return None
        if self._contains_any(lowered_question, ["count", "number of", "total number"]) and not self._contains_any(lowered_question, ["average", "avg"]):
            return None

        metric_keywords = [
            token
            for token in ["grade", "grades", "score", "scores", "mark", "marks", "rating", "ratings", "value", "values"]
            if token in lowered_question
        ]

        metric_table = None
        if metric_keywords:
            metric_table = self._find_table_by_keywords(metadata, [metric_keywords])
        elif self._contains_any(lowered_question, ["performance", "performing", "rank", "ranking", "top", "highest", "lowest", "average", "avg", "second"]):
            metric_table = self._find_table_by_keywords(metadata, [["grade", "score", "rating", "mark", "points", "value"]])

        if not metric_table and context.relevant_tables:
            for candidate in context.relevant_tables:
                if self._pick_numeric_column(metadata, candidate, lowered_question):
                    metric_table = candidate
                    break

        if not metric_table:
            metric_table = self._choose_best_table(
                lowered_question,
                metadata,
                context,
                prioritize_metric_table=True,
                intent=intent,
                role_index=role_index,
            )

        metric_col = self._pick_numeric_column(metadata, metric_table, lowered_question) if metric_table else None
        if metric_table and metric_keywords:
            preferred_metric = self._find_column_by_keywords(metadata, metric_table, metric_keywords)
            if preferred_metric:
                metric_col = preferred_metric

        top_n = intent.top_n if intent and intent.top_n else 5
        if top_n == 5 and self._contains_any(lowered_question, ["highest", "lowest", "best", "worst"]) and not re.search(r"\btop\s+\d+\b", lowered_question):
            top_n = 1

        if ("top-performing" in lowered_question or "top performing" in lowered_question) and self._contains_any(lowered_question, ["each", "per"]):
            entity_table = self._find_table_by_keywords(metadata, [["student", "students", "learner", "learners", "employee", "employees"]])
            group_table = None
            if self._contains_any(lowered_question, ["course", "courses", "class", "classes", "subject", "subjects"]):
                group_table = self._find_table_by_keywords(metadata, [["course", "courses", "class", "classes", "subject", "subjects"]])
            if not group_table:
                group_table = self._find_table_by_keywords(metadata, [["department", "departments", "team", "teams", "group", "groups"]])

            if entity_table and group_table:
                perf_table = None
                perf_col = None
                fk_perf_to_entity = None
                fk_perf_to_group = None
                for table_name in metadata.schema.get("tables", {}).keys():
                    candidate_col = self._find_column_by_keywords(
                        metadata,
                        table_name,
                        ["grade", "score", "rating", "mark", "points", "gpa"],
                    )
                    if not candidate_col:
                        continue
                    fk_entity = self._find_fk_to_table(metadata, table_name, entity_table)
                    fk_group = self._find_fk_to_table(metadata, table_name, group_table)
                    if fk_entity and fk_group:
                        perf_table = table_name
                        perf_col = candidate_col
                        fk_perf_to_entity = fk_entity
                        fk_perf_to_group = fk_group
                        break

                if perf_table and perf_col and fk_perf_to_entity and fk_perf_to_group:
                    entity_id = self._find_table_id_column(metadata, entity_table)
                    entity_label = self._pick_name_like_column(metadata, entity_table) or entity_id
                    group_id = self._find_table_id_column(metadata, group_table)
                    group_label = self._pick_name_like_column(metadata, group_table) or group_id
                    if all([entity_id, entity_label, group_id, group_label]):
                        return (
                            f"SELECT * FROM ("
                            f"SELECT e.{entity_label}, g.{group_label} AS course, "
                            f"RANK() OVER (PARTITION BY g.{group_id} ORDER BY m.{perf_col} DESC) AS rnk "
                            f"FROM {entity_table} e "
                            f"JOIN {perf_table} m ON e.{entity_id} = m.{fk_perf_to_entity} "
                            f"JOIN {group_table} g ON m.{fk_perf_to_group} = g.{group_id}"
                            f") t WHERE rnk = 1"
                        )

        if "average" in lowered_question and self._contains_any(lowered_question, ["top", "highest"]):
            entity_phrase_match = re.search(r"top\s+\d+\s+([a-z_]+)", lowered_question)
            entity_table = None
            if entity_phrase_match:
                entity_candidates = self._candidate_tables_from_phrase(metadata, entity_phrase_match.group(1))
                if entity_candidates:
                    entity_table = entity_candidates[0]

            if not entity_table and context.relevant_tables:
                for candidate in context.relevant_tables:
                    if candidate != metric_table and self._table_mentioned_in_question(lowered_question, candidate):
                        entity_table = candidate
                        break

            if not entity_table:
                entity_table = self._find_table_by_keywords(
                    metadata,
                    [["student", "students", "learner", "learners", "user", "users"]],
                )

            if entity_table and metric_table and metric_col:
                entity_id = self._find_table_id_column(metadata, entity_table)
                entity_label = self._pick_name_like_column(metadata, entity_table) or entity_id
                fk_metric_to_entity = self._find_fk_to_table(metadata, metric_table, entity_table)
                fk_entity_to_metric = self._find_fk_to_table(metadata, entity_table, metric_table)
                metric_id = self._find_table_id_column(metadata, metric_table)

                if entity_id and entity_label and fk_metric_to_entity:
                    entity_alias = f"{entity_table[:-1]}_name" if entity_table.endswith("s") else f"{entity_table}_name"
                    metric_alias = f"{metric_col.lower()}_average"
                    return (
                        f"SELECT e.{entity_label} AS {entity_alias}, AVG(m.{metric_col}) AS {metric_alias} "
                        f"FROM {entity_table} e LEFT JOIN {metric_table} m ON e.{entity_id} = m.{fk_metric_to_entity} "
                        f"GROUP BY e.{entity_label} ORDER BY {metric_alias} DESC LIMIT {top_n}"
                    )

                if entity_id and entity_label and fk_entity_to_metric and metric_id:
                    entity_alias = f"{entity_table[:-1]}_name" if entity_table.endswith("s") else f"{entity_table}_name"
                    metric_alias = f"{metric_col.lower()}_average"
                    return (
                        f"SELECT e.{entity_label} AS {entity_alias}, AVG(m.{metric_col}) AS {metric_alias} "
                        f"FROM {entity_table} e LEFT JOIN {metric_table} m ON e.{fk_entity_to_metric} = m.{metric_id} "
                        f"GROUP BY e.{entity_label} ORDER BY {metric_alias} DESC LIMIT {top_n}"
                    )

        if "second" in lowered_question and "highest" in lowered_question and self._contains_any(lowered_question, ["each", "per", "in each"]):
            group_phrase_match = re.search(r"(?:in\s+each|for\s+each|each|per)\s+([a-z_]+)", lowered_question)
            group_table = None
            if group_phrase_match:
                group_candidates = self._candidate_tables_from_phrase(metadata, group_phrase_match.group(1))
                if group_candidates:
                    group_table = group_candidates[0]

            if not group_table and context.relevant_tables:
                for candidate in context.relevant_tables:
                    if candidate != metric_table and self._table_mentioned_in_question(lowered_question, candidate):
                        group_table = candidate
                        break

            if group_table and metric_table and metric_col:
                group_id = self._find_table_id_column(metadata, group_table)
                group_label = self._pick_name_like_column(metadata, group_table) or group_id
                fk_metric_to_group = self._find_fk_to_table(metadata, metric_table, group_table)

                if group_id and group_label and fk_metric_to_group:
                    return (
                        f"SELECT * FROM ("
                        f"SELECT g.{group_label}, m.{metric_col}, "
                        f"DENSE_RANK() OVER (PARTITION BY g.{group_id} ORDER BY m.{metric_col} DESC) AS rnk "
                        f"FROM {group_table} g JOIN {metric_table} m ON m.{fk_metric_to_group} = g.{group_id}"
                        f") t WHERE rnk = 2"
                    )

        if "rank" in lowered_question and self._contains_any(lowered_question, ["within", "each", "per"]) and self._contains_any(lowered_question, ["performance", "grade", "score"]):
            entity_table = self._find_table_by_keywords(metadata, [["student", "students", "learner", "learners", "employee", "employees"]])
            group_table = self._find_table_by_keywords(metadata, [["department", "departments", "team", "teams", "group", "groups"]])
            if entity_table and group_table and metric_table and metric_col:
                entity_id = self._find_table_id_column(metadata, entity_table)
                entity_label = self._pick_name_like_column(metadata, entity_table) or entity_id
                group_id = self._find_table_id_column(metadata, group_table)
                group_label = self._pick_name_like_column(metadata, group_table) or group_id
                fk_entity_to_group = self._find_fk_to_table(metadata, entity_table, group_table)
                fk_metric_to_entity = self._find_fk_to_table(metadata, metric_table, entity_table)

                if all([entity_id, entity_label, group_id, group_label, fk_entity_to_group, fk_metric_to_entity]):
                    return (
                        f"SELECT e.{entity_label}, e.{fk_entity_to_group}, "
                        f"RANK() OVER (PARTITION BY e.{fk_entity_to_group} ORDER BY AVG(m.{metric_col}) DESC) AS rank "
                        f"FROM {entity_table} e "
                        f"JOIN {metric_table} m ON m.{fk_metric_to_entity} = e.{entity_id} "
                        f"GROUP BY e.{entity_id}"
                    )

        if ("top-performing" in lowered_question or "top performing" in lowered_question) and self._contains_any(lowered_question, ["each", "per"]):
            entity_table = self._find_table_by_keywords(metadata, [["student", "students", "learner", "learners"]])
            group_table = self._find_table_by_keywords(metadata, [["course", "courses", "department", "departments"]])
            if entity_table and group_table:
                if (
                    not metric_table
                    or not metric_col
                    or metric_table in {entity_table, group_table}
                    or not any(token in metric_col.lower() for token in ["grade", "score", "rating", "mark", "points", "gpa"])
                ):
                    candidate_metric_table = None
                    candidate_metric_col = None
                    for table_name in metadata.schema.get("tables", {}).keys():
                        if table_name in {entity_table, group_table}:
                            continue
                        perf_col = self._find_column_by_keywords(
                            metadata,
                            table_name,
                            ["grade", "score", "rating", "mark", "points", "gpa"],
                        )
                        if not perf_col:
                            continue
                        fk_to_entity = self._find_fk_to_table(metadata, table_name, entity_table)
                        fk_to_group = self._find_fk_to_table(metadata, table_name, group_table)
                        if fk_to_entity and fk_to_group:
                            candidate_metric_table = table_name
                            candidate_metric_col = perf_col
                            break

                    if candidate_metric_table and candidate_metric_col:
                        metric_table = candidate_metric_table
                        metric_col = candidate_metric_col

            if entity_table and group_table and metric_table and metric_col:
                entity_id = self._find_table_id_column(metadata, entity_table)
                entity_label = self._pick_name_like_column(metadata, entity_table) or entity_id
                group_id = self._find_table_id_column(metadata, group_table)
                group_label = self._pick_name_like_column(metadata, group_table) or group_id
                fk_metric_to_entity = self._find_fk_to_table(metadata, metric_table, entity_table)
                fk_metric_to_group = self._find_fk_to_table(metadata, metric_table, group_table)

                if all([entity_id, entity_label, group_id, group_label, fk_metric_to_entity, fk_metric_to_group]):
                    return (
                        f"SELECT * FROM ("
                        f"SELECT e.{entity_label}, g.{group_label} AS course, "
                        f"RANK() OVER (PARTITION BY g.{group_id} ORDER BY m.{metric_col} DESC) AS rnk "
                        f"FROM {entity_table} e "
                        f"JOIN {metric_table} m ON e.{entity_id} = m.{fk_metric_to_entity} "
                        f"JOIN {group_table} g ON m.{fk_metric_to_group} = g.{group_id}"
                        f") t WHERE rnk = 1"
                    )

        # Count enrolled students per course (or equivalent group/entity phrasing).
        if self._contains_any(lowered_question, ["number of", "count"]) and self._contains_any(lowered_question, ["enrolled"]):
            group_table = self._find_table_by_keywords(metadata, [["course", "courses", "class", "classes", "subject", "subjects"]])
            entity_table = self._find_table_by_keywords(metadata, [["student", "students", "learner", "learners"]])
            if group_table and entity_table:
                group_id = self._find_table_id_column(metadata, group_table)
                group_label = self._pick_name_like_column(metadata, group_table) or group_id
                if group_id and group_label:
                    bridge = self._find_bridge_table(metadata, group_table, entity_table)
                    if bridge:
                        bridge_table, fk_group, fk_entity = bridge
                        return (
                            f"SELECT g.{group_label}, COUNT(b.{fk_entity}) AS enrolled_students "
                            f"FROM {group_table} g LEFT JOIN {bridge_table} b ON g.{group_id} = b.{fk_group} "
                            f"GROUP BY g.{group_id}, g.{group_label}"
                        )

        join_info = self._find_join_pair(metadata, context.relevant_tables)
        if not join_info:
            return None

        child_table, parent_table, child_fk, parent_pk = join_info
        child_metric = self._pick_numeric_column(metadata, child_table, lowered_question)
        if not child_metric:
            child_metric = self._pick_numeric_column(metadata, parent_table, lowered_question)
            if child_metric:
                child_table, parent_table = parent_table, child_table
                child_fk, parent_pk = parent_pk, child_fk
        if not child_metric:
            return None

        child_label = self._pick_name_like_column(metadata, child_table) or self._find_table_id_column(metadata, child_table)
        parent_label = self._pick_name_like_column(metadata, parent_table) or parent_pk
        if not child_label:
            return None

        top_n = intent.top_n if intent and intent.top_n else 5

        if self._contains_any(lowered_question, ["top-performing", "top performing", "performance", "performing"]) and not self._contains_any(str(child_metric).lower(), ["grade", "score", "rating", "mark", "points", "gpa"]):
            return None

        if "second" in lowered_question and "highest" in lowered_question:
            return (
                f"SELECT grouped_label, metric_value FROM ("
                f"SELECT p.{parent_label} AS grouped_label, c.{child_metric} AS metric_value, "
                f"DENSE_RANK() OVER (PARTITION BY p.{parent_pk} ORDER BY c.{child_metric} DESC) AS ranking_value "
                f"FROM {parent_table} p JOIN {child_table} c ON c.{child_fk} = p.{parent_pk}"
                f") ranked WHERE ranking_value = 2"
            )

        if "rank" in lowered_question and self._contains_any(lowered_question, ["within", "each", "per"]):
            return (
                f"SELECT entity_label, grouped_label, metric_value, ranking_value FROM ("
                f"SELECT c.{child_label} AS entity_label, p.{parent_label} AS grouped_label, c.{child_metric} AS metric_value, "
                f"ROW_NUMBER() OVER (PARTITION BY p.{parent_pk} ORDER BY c.{child_metric} DESC) AS ranking_value "
                f"FROM {child_table} c JOIN {parent_table} p ON c.{child_fk} = p.{parent_pk}"
                f") ranked"
            )

        if "average" in lowered_question and self._contains_any(lowered_question, ["highest", "top"]):
            return (
                f"SELECT p.{parent_label}, AVG(c.{child_metric}) AS average_value "
                f"FROM {parent_table} p JOIN {child_table} c ON c.{child_fk} = p.{parent_pk} "
                f"GROUP BY p.{parent_pk}, p.{parent_label} ORDER BY average_value DESC LIMIT {top_n}"
            )

        if self._contains_any(lowered_question, ["top-performing", "top performing"]) and self._contains_any(lowered_question, ["each", "per"]):
            return (
                f"SELECT grouped_label, entity_label, metric_value FROM ("
                f"SELECT p.{parent_label} AS grouped_label, c.{child_label} AS entity_label, c.{child_metric} AS metric_value, "
                f"ROW_NUMBER() OVER (PARTITION BY p.{parent_pk} ORDER BY c.{child_metric} DESC) AS rn "
                f"FROM {parent_table} p JOIN {child_table} c ON c.{child_fk} = p.{parent_pk}"
                f") ranked WHERE rn = 1"
            )

        if self._contains_any(lowered_question, ["top", "highest", "most"]):
            return (
                f"SELECT c.{child_label}, AVG(c.{child_metric}) AS average_value "
                f"FROM {child_table} c GROUP BY c.{child_label} ORDER BY average_value DESC LIMIT {top_n}"
            )

        return None


    def repair(
        self,
        question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
        error_message: str,
        previous_sql: str,
    ) -> str:
        # Targeted repair strategy for first-phase implementation.
        if "no such table" in error_message.lower() and context.relevant_tables:
            table = context.relevant_tables[0]
            return previous_sql.replace(self._extract_from_table(previous_sql), table)

        if "no such column" in error_message.lower():
            table = self._choose_best_table(question.lower(), metadata, context)
            columns = self._choose_projection_columns(question.lower(), metadata, table, context)
            if columns:
                return f"SELECT {', '.join(columns[:5])} FROM {table}"

        return self.generate(question, metadata, context)

    def _choose_best_table(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
        prioritize_metric_table: bool = False,
        intent: QueryIntent | None = None,
        role_index: SchemaRoleIndex | None = None,
    ) -> str:
        candidates = context.relevant_tables or list(metadata.schema["tables"].keys())
        if not candidates:
            return ""

        role_index = role_index or self._infer_schema_roles(metadata)

        # Prefer entity tables when classifier identifies entities from semantic intent.
        if intent and intent.inferred_entities:
            for entity_term in intent.inferred_entities:
                for table in candidates:
                    if self._contains_any(table.lower(), [entity_term]):
                        return table

        # If we need to prioritize the table with metric columns for aggregations.
        if prioritize_metric_table or (intent and intent.needs_aggregation):
            # Prefer tables where inferred metrics are present.
            if intent and intent.inferred_metrics:
                for table in candidates:
                    metric_cols = [c.lower() for c in role_index.metric_columns.get(table, [])]
                    if any(any(metric in col for col in metric_cols) for metric in intent.inferred_metrics):
                        return table

            # Fall back to any candidate with numeric/metric columns.
            for table in candidates:
                if role_index.metric_columns.get(table):
                    return table
                if self._pick_numeric_column(metadata, table, lowered_question):
                    return table

        scored = sorted(
            ((self._score_table(lowered_question, table, metadata), table) for table in candidates),
            reverse=True,
        )
        return scored[0][1]

    def _choose_projection_columns(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        table: str,
        context: RetrievalContext,
    ) -> list[str]:
        if not table:
            return []

        available = [col["name"] for col in metadata.schema["tables"].get(table, {}).get("columns", [])]
        if not available:
            return []

        scored = sorted(
            ((self._score_column(lowered_question, table, col, metadata), col) for col in available),
            reverse=True,
        )

        chosen = [col for score, col in scored if score > 0]
        if not chosen:
            chosen = context.relevant_columns.get(table, []) or available[:5]

        name_only_request = (
            self._contains_any(lowered_question, ["names of", "name of"]) 
            and self._contains_any(lowered_question, ["all", "list", "show", "display", "retrieve"]) 
        )
        if name_only_request:
            name_like = self._pick_name_like_column(metadata, table)
            if name_like:
                return [name_like]

        # For list/show style queries, include an entity label column instead of only numeric attributes.
        label_col = self._pick_name_like_column(metadata, table)
        if label_col and self._contains_any(lowered_question, ["list", "show", "find", "retrieve", "display"]):
            if label_col not in chosen:
                chosen = [label_col] + chosen
            elif chosen[0] != label_col:
                chosen = [label_col] + [c for c in chosen if c != label_col]

        return chosen[:5]

    def _pick_numeric_column(self, metadata: DatabaseMetadata, table: str, question: str = "") -> str | None:
        table_info = metadata.schema["tables"].get(table, {})
        columns = table_info.get("columns", [])
        
        numeric_columns: list[tuple[float, str]] = []
        lowered_question = question.lower()
        question_has_status_intent = self._contains_any(lowered_question, ["active", "status", "current", "enabled", "flag"])
        question_has_metric_intent = self._contains_any(
            lowered_question,
            ["average", "avg", "performance", "rank", "top", "highest", "lowest", "second", "grade", "score", "mark", "rating", "value", "amount"],
        )
        
        for col in columns:
            original_name = str(col.get("name", ""))
            col_name = original_name.lower()
            col_type = str(col.get("type", "")).lower()
            
            # Check if column type is numeric
            if not any(token in col_type for token in ["int", "real", "num", "decimal", "float"]):
                continue
            
            # Skip ID columns
            if "id" in col_name:
                continue
            
            # Score using semantic relevance if question provided.
            score = 0.0
            if question:
                score = self._score_column_relevance(lowered_question, col_name)

                if any(token in col_name for token in ["grade", "score", "mark", "rating", "point", "points", "value", "amount", "total", "credit", "credits"]):
                    score += 2.5

                if any(token in col_name for token in ["status", "state", "active", "enabled", "flag", "is_active"]):
                    if question_has_status_intent:
                        score += 2.0
                    else:
                        score -= 4.0

                if question_has_metric_intent and any(token in col_name for token in ["grade", "score", "mark", "rating", "point", "points", "value", "amount", "total", "credit", "credits"]):
                    score += 2.0

                if question_has_metric_intent and any(token in col_name for token in ["status", "state", "active", "enabled", "flag", "is_active"]):
                    score -= 2.5

            numeric_columns.append((score, original_name))
        
        if numeric_columns:
            numeric_columns.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return numeric_columns[0][1]
        
        for column in columns:
            col_type = str(column.get("type", "")).lower()
            if any(token in col_type for token in ["int", "real", "num", "decimal", "float"]):
                return column["name"]
        
        return None

    def _pick_text_column(self, metadata: DatabaseMetadata, table: str) -> str | None:
        table_info = metadata.schema["tables"].get(table, {})
        for column in table_info.get("columns", []):
            col_type = str(column.get("type", "")).lower()
            if "char" in col_type or "text" in col_type:
                return column["name"]
        return None

    def _extract_from_table(self, sql: str) -> str:
        tokens = sql.split()
        for idx, token in enumerate(tokens):
            if token.upper() == "FROM" and idx + 1 < len(tokens):
                return tokens[idx + 1]
        return ""

    def _try_generate_join_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
        intent: QueryIntent | None = None,
        role_index: SchemaRoleIndex | None = None,
    ) -> str | None:
        role_index = role_index or self._infer_schema_roles(metadata)
        mentioned_tables = self._mentioned_tables(lowered_question, context.relevant_tables)
        primary_table = self._primary_table_from_question(lowered_question, metadata, context)

        if primary_table and len(mentioned_tables) >= 2 and " and " in lowered_question:
            other_table = next((t for t in mentioned_tables if t != primary_table), None)
            if other_table:
                primary_id = self._find_table_id_column(metadata, primary_table)
                other_id = self._find_table_id_column(metadata, other_table)
                primary_label = self._pick_name_like_column(metadata, primary_table) or primary_id
                other_label = self._pick_name_like_column(metadata, other_table) or other_id
                if primary_id and other_id and primary_label and other_label:
                    fk_primary_to_other = self._find_fk_to_table(metadata, primary_table, other_table)
                    if fk_primary_to_other:
                        return (
                            f"SELECT p.{primary_label}, o.{other_label} "
                            f"FROM {primary_table} p JOIN {other_table} o ON p.{fk_primary_to_other} = o.{other_id}"
                        )

                    fk_other_to_primary = self._find_fk_to_table(metadata, other_table, primary_table)
                    if fk_other_to_primary:
                        return (
                            f"SELECT p.{primary_label}, o.{other_label} "
                            f"FROM {primary_table} p JOIN {other_table} o ON o.{fk_other_to_primary} = p.{primary_id}"
                        )

        if len(mentioned_tables) <= 1 and self._contains_any(lowered_question, ["more than", "less than", "greater than", "at least", "at most"]):
            return None

        # Early exit for clearly non-join intents unless multiple tables are strongly relevant.
        if intent and not intent.needs_join and len(context.relevant_tables) < 2:
            return None

        # Avoid joins when question asks for attributes already present in one mentioned table.
        if any(hint in lowered_question for hint in ["along with", "with their", "including", "together with"]):
            mentioned_tables = [
                table
                for table in (context.relevant_tables or [])
                if self._table_mentioned_in_question(lowered_question, table)
            ]
            if len(set(mentioned_tables)) <= 1 and context.relevant_tables:
                main_table = mentioned_tables[0] if mentioned_tables else context.relevant_tables[0]
                main_cols = [
                    str(col.get("name", "")).lower()
                    for col in metadata.schema.get("tables", {}).get(main_table, {}).get("columns", [])
                ]
                requested_attributes = [
                    token
                    for token in ["country", "email", "address", "phone", "city", "state", "postal", "zip"]
                    if token in lowered_question
                ]
                if requested_attributes and all(
                    any(attr in col for col in main_cols) for attr in requested_attributes
                ):
                    return None

        # Handle "along with" patterns using schema relationships and question-mentioned tables.
        if any(hint in lowered_question for hint in ["along with", "with their", "with the", "including", "together with"]):
            if context.relevant_tables and len(context.relevant_tables) >= 2:
                main_table = context.relevant_tables[0]
                main_table_info = metadata.schema["tables"].get(main_table, {})
                foreign_keys = main_table_info.get("foreign_keys", [])
                select_cols = []
                from_clause = f" FROM {main_table} t1"
                alias_counter = 2
                aliases: dict[str, str] = {main_table: "t1"}

                name_col = self._pick_name_like_column(metadata, main_table)
                if name_col:
                    select_cols.append(f"t1.{name_col}")

                # Prefer only tables explicitly mentioned in the question text.
                mentioned_targets = {
                    table
                    for table in context.relevant_tables[1:]
                    if self._table_mentioned_in_question(lowered_question, table)
                }

                for fk in foreign_keys:
                    parent_table = fk.get("references_table")
                    fk_col = fk.get("column")
                    ref_col = fk.get("references_column")

                    if parent_table and parent_table in mentioned_targets:
                        parent_name_col = self._pick_name_like_column(metadata, parent_table)
                        alias = f"t{alias_counter}"
                        alias_counter += 1

                        if parent_name_col:
                            select_cols.append(f"{alias}.{parent_name_col}")
                            from_clause += f" JOIN {parent_table} {alias} ON t1.{fk_col} = {alias}.{ref_col}"
                            aliases[parent_table] = alias

                # Add one more hop for mentioned tables not directly related to main table.
                for joined_table, joined_alias in list(aliases.items()):
                    joined_info = metadata.schema["tables"].get(joined_table, {})
                    for fk in joined_info.get("foreign_keys", []):
                        parent_table = fk.get("references_table")
                        fk_col = fk.get("column")
                        ref_col = fk.get("references_column")
                        if (
                            parent_table
                            and parent_table in mentioned_targets
                            and parent_table not in aliases
                            and fk_col
                            and ref_col
                        ):
                            parent_name_col = self._pick_name_like_column(metadata, parent_table)
                            if parent_name_col:
                                alias = f"t{alias_counter}"
                                alias_counter += 1
                                select_cols.append(f"{alias}.{parent_name_col}")
                                from_clause += f" JOIN {parent_table} {alias} ON {joined_alias}.{fk_col} = {alias}.{ref_col}"
                                aliases[parent_table] = alias

                if len(select_cols) > 1:
                    return f"SELECT {', '.join(select_cols)}{from_clause}"

        # Generic many-to-many listing via bridge table when direct join is unavailable.
        if any(hint in lowered_question for hint in ["along with", "with their", "including", "together with"]) and len(context.relevant_tables) >= 2:
            t1, t2 = context.relevant_tables[0], context.relevant_tables[1]
            bridge = self._find_bridge_table(metadata, t1, t2)
            if bridge:
                bridge_table, fk_t1, fk_t2 = bridge
                t1_id = self._find_table_id_column(metadata, t1)
                t2_id = self._find_table_id_column(metadata, t2)
                t1_name = self._pick_name_like_column(metadata, t1)
                t2_name = self._pick_name_like_column(metadata, t2)
                if all([t1_id, t2_id, t1_name, t2_name]):
                    return (
                        f"SELECT a.{t1_name}, b.{t2_name} "
                        f"FROM {t1} a "
                        f"JOIN {bridge_table} x ON a.{t1_id} = x.{fk_t1} "
                        f"JOIN {t2} b ON x.{fk_t2} = b.{t2_id}"
                    )

        # Generic "X teach more than N Y" style counts.
        teach_match = re.search(r"more than\s+(\d+)", lowered_question)
        if teach_match and self._contains_any(lowered_question, ["teach", "teaches", "teaching"]):
            threshold = int(teach_match.group(1))
            teacher_table = self._find_table_by_keywords(metadata, [["instructor", "teacher", "professor", "faculty"]])
            taught_table = self._find_table_by_keywords(metadata, [["course", "class", "subject"]])
            if teacher_table and taught_table:
                fk_taught_to_teacher = self._find_fk_to_table(metadata, taught_table, teacher_table)
                teacher_id = self._find_table_id_column(metadata, teacher_table)
                teacher_label = self._pick_name_like_column(metadata, teacher_table) or teacher_id
                taught_id = self._find_table_id_column(metadata, taught_table)
                if fk_taught_to_teacher and teacher_id and teacher_label and taught_id:
                    return (
                        f"SELECT t.{teacher_label} "
                        f"FROM {teacher_table} t JOIN {taught_table} c ON t.{teacher_id} = c.{fk_taught_to_teacher} "
                        f"GROUP BY t.{teacher_id} HAVING COUNT(c.{taught_id}) > {threshold}"
                    )

        enroll_match = re.search(r"more than\s+(\d+)", lowered_question)
        if enroll_match and self._contains_any(lowered_question, ["enroll", "enrolled", "enrollment"]) and self._contains_any(lowered_question, ["course", "courses"]):
            threshold = int(enroll_match.group(1))
            student_table = self._find_table_by_keywords(metadata, [["student", "students", "learner", "learners"]])
            course_table = self._find_table_by_keywords(metadata, [["course", "courses", "class", "classes", "subject", "subjects"]])
            if student_table and course_table:
                bridge = self._find_bridge_table(metadata, student_table, course_table)
                if bridge:
                    bridge_table, fk_student, fk_course = bridge
                    student_id = self._find_table_id_column(metadata, student_table)
                    student_label = self._pick_name_like_column(metadata, student_table) or student_id
                    if student_id and student_label:
                        return (
                            f"SELECT s.{student_label} "
                            f"FROM {student_table} s "
                            f"JOIN {bridge_table} e ON s.{student_id} = e.{fk_student} "
                            f"GROUP BY s.{student_id}, s.{student_label} HAVING COUNT(e.{fk_course}) > {threshold}"
                        )

        # Generic latest-date lookup.
        if any(term in lowered_question for term in ["latest", "most recent", "newest"]):
            latest_table = self._choose_best_table(lowered_question, metadata, context, prioritize_metric_table=True)
            if latest_table:
                date_col = self._find_column_by_keywords(metadata, latest_table, ["date", "time", "created_at", "updated_at", "timestamp"])
                if date_col:
                    select_cols = [str(col.get("name", "")) for col in metadata.schema.get("tables", {}).get(latest_table, {}).get("columns", [])][:5]
                    select_clause = ", ".join(select_cols) if select_cols else "*"
                    return f"SELECT {select_clause} FROM {latest_table} ORDER BY {date_col} DESC LIMIT 1"

        # Generic "entity working on more than N related rows" pattern via bridge table.
        count_match = re.search(r"more than\s+(\d+)", lowered_question)
        if count_match and len(context.relevant_tables) >= 2:
            threshold = int(count_match.group(1))
            t1, t2 = context.relevant_tables[0], context.relevant_tables[1]
            bridge = self._find_bridge_table(metadata, t1, t2)
            if bridge:
                bridge_table, fk_t1, fk_t2 = bridge
                t1_id = self._find_table_id_column(metadata, t1)
                t1_name = self._pick_name_like_column(metadata, t1)
                if all([t1_id, t1_name, fk_t1, fk_t2]):
                    return (
                        f"SELECT a.{t1_name} "
                        f"FROM {t1} a "
                        f"JOIN {bridge_table} x ON a.{t1_id} = x.{fk_t1} "
                        f"GROUP BY a.{t1_id}, a.{t1_name} "
                        f"HAVING COUNT(x.{fk_t2}) > {threshold}"
                    )

        # Generic "which X has the most Y" pattern based on counting related rows.
        if any(term in lowered_question for term in ["most", "highest", "least", "lowest"]):
            join_info = self._find_join_pair(metadata, context.relevant_tables)
            if join_info:
                child_table, parent_table, child_col, parent_col = join_info
                parent_label = self._pick_name_like_column(metadata, parent_table) or parent_col
                child_pk = self._find_column_by_keywords(metadata, child_table, ["id", f"{child_table[:-1]}_id"]) or child_col
                sort_dir = "DESC" if any(term in lowered_question for term in ["most", "highest"]) else "ASC"
                return (
                    f"SELECT p.{parent_label} "
                    f"FROM {parent_table} p JOIN {child_table} c ON c.{child_col} = p.{parent_col} "
                    f"GROUP BY p.{parent_col} ORDER BY COUNT(c.{child_pk}) {sort_dir} LIMIT 1"
                )

        # Generic anti-join pattern: parent entities with no related children.
        if any(term in lowered_question for term in ["no ", "without", "does not have", "do not have", "have no"]):
            if primary_table and len(mentioned_tables) >= 2:
                detail_table = next((t for t in mentioned_tables if t != primary_table), None)
                if detail_table:
                    primary_id = self._find_table_id_column(metadata, primary_table)
                    primary_label = self._pick_name_like_column(metadata, primary_table) or primary_id
                    bridge = self._find_bridge_table(metadata, primary_table, detail_table)
                    if bridge and primary_id and primary_label:
                        bridge_table, fk_primary, fk_detail = bridge
                        return (
                            f"SELECT p.{primary_label} "
                            f"FROM {primary_table} p LEFT JOIN {bridge_table} b ON p.{primary_id} = b.{fk_primary} "
                            f"WHERE b.{fk_detail} IS NULL"
                        )

            join_info = self._find_join_pair(metadata, context.relevant_tables)
            if join_info:
                child_table, parent_table, child_col, parent_col = join_info
                parent_label = self._pick_name_like_column(metadata, parent_table) or parent_col
                child_pk = self._find_column_by_keywords(metadata, child_table, ["id", f"{child_table[:-1]}_id"]) or child_col
                return (
                    f"SELECT p.{parent_label} "
                    f"FROM {parent_table} p LEFT JOIN {child_table} c ON c.{child_col} = p.{parent_col} "
                    f"WHERE c.{child_pk} IS NULL"
                )

        # Generic "belongs to groups with more than N members" pattern.
        threshold_match = re.search(r"more than\s+(\d+)", lowered_question)
        if threshold_match and any(term in lowered_question for term in ["belong", "belongs", "belonging"]):
            join_info = self._find_join_pair(metadata, context.relevant_tables)
            if join_info:
                child_table, _, child_col, _ = join_info
                child_label = self._pick_name_like_column(metadata, child_table)
                if child_label:
                    threshold = int(threshold_match.group(1))
                    return (
                        f"SELECT {child_label} FROM {child_table} "
                        f"WHERE {child_col} IN ("
                        f"SELECT {child_col} FROM {child_table} GROUP BY {child_col} HAVING COUNT(*) > {threshold}"
                        f")"
                    )

        # Generic top-N per group window pattern.
        if self._contains_any(lowered_question, ["top", "highest", "oldest", "youngest"]) and self._contains_any(lowered_question, ["each", "per"]):
            join_info = self._find_join_pair(metadata, context.relevant_tables)
            if join_info:
                child_table, _, child_group_fk, _ = join_info
                child_label = self._pick_name_like_column(metadata, child_table)
                metric_col = self._pick_numeric_column(metadata, child_table, lowered_question)
                n = self._extract_top_n(lowered_question, default_value=2)
                if child_label and metric_col:
                    direction = "ASC" if self._contains_any(lowered_question, ["youngest", "lowest"]) else "DESC"
                    return (
                        f"SELECT {child_label}, {metric_col} FROM ("
                        f"SELECT {child_label}, {metric_col}, "
                        f"ROW_NUMBER() OVER (PARTITION BY {child_group_fk} ORDER BY {metric_col} {direction}) AS rn "
                        f"FROM {child_table}"
                        f") WHERE rn <= {n}"
                    )

        join_hints = ["along with", "together with", "including", "with their", "with the"]
        non_listing_join_intents = [
            "average", "count", "number", "top", "highest", "lowest", "most", "least",
            "more than", "hours", "worked", "budget", "performance", "rating", "absent",
            "latest", "oldest", "youngest", "working on", "belong", "teach", "teaches", "teaching",
        ]
        if re.search(r"\bfrom\s+[a-z]+\b", lowered_question) and self._contains_any(lowered_question, ["email", "country", "address", "phone"]):
            return None
        if not any(hint in lowered_question for hint in join_hints) and not any(
            token in lowered_question for token in non_listing_join_intents
        ):
            return None

        tables = mentioned_tables if len(mentioned_tables) >= 2 else context.relevant_tables
        if len(tables) < 2:
            return None

        join_info = self._find_join_pair(metadata, tables)
        if join_info is None:
            return None

        child_table, parent_table, child_col, parent_col = join_info
        child_name_col = self._pick_name_like_column(metadata, child_table) or self._find_table_id_column(metadata, child_table)
        parent_name_col = self._pick_name_like_column(metadata, parent_table) or self._find_table_id_column(metadata, parent_table)
        if not child_name_col or not parent_name_col:
            return None

        return (
            f"SELECT c.{child_name_col}, p.{parent_name_col} "
            f"FROM {child_table} c "
            f"JOIN {parent_table} p ON c.{child_col} = p.{parent_col}"
        )

    def _find_join_pair(
        self,
        metadata: DatabaseMetadata,
        tables: list[str],
    ) -> tuple[str, str, str, str] | None:
        # Prefer explicit foreign key metadata when available.
        for child_table in tables:
            child_info = metadata.schema["tables"].get(child_table, {})
            for fk in child_info.get("foreign_keys", []):
                parent_table = fk.get("references_table")
                if parent_table in tables:
                    return (
                        child_table,
                        parent_table,
                        fk.get("column", ""),
                        fk.get("references_column", ""),
                    )

        # Heuristic fallback for databases that omit FK constraints:
        # infer joins from columns like <table>_id -> <table>.id.
        table_data = metadata.schema.get("tables", {})
        for child_table in tables:
            child_cols = [
                str(col.get("name", "")).lower()
                for col in table_data.get(child_table, {}).get("columns", [])
            ]
            for parent_table in tables:
                if parent_table == child_table:
                    continue

                parent_cols = [
                    str(col.get("name", "")).lower()
                    for col in table_data.get(parent_table, {}).get("columns", [])
                ]
                parent_singular = parent_table[:-1] if parent_table.endswith("s") else parent_table
                parent_key_candidates = [
                    "id",
                    f"{parent_singular}_id",
                ]
                parent_key_candidates.extend(
                    [col for col in parent_cols if col.endswith("_id")]
                )

                parent_key = None
                for key in parent_key_candidates:
                    if key in parent_cols:
                        parent_key = key
                        break

                if parent_key is None:
                    continue

                candidate_child_cols = {
                    f"{parent_table}_id",
                    f"{parent_singular}_id",
                }
                if parent_key != "id":
                    candidate_child_cols.add(parent_key)
                for candidate in candidate_child_cols:
                    if candidate in child_cols:
                        return (child_table, parent_table, candidate, parent_key)
        return None

    def _pick_name_like_column(self, metadata: DatabaseMetadata, table: str) -> str | None:
        table_info = metadata.schema["tables"].get(table, {})
        columns = table_info.get("columns", [])

        # Prefer generic "name" column
        for column in columns:
            name = str(column.get("name", "")).lower()
            if name == "name":
                return column["name"]

        # Then check for columns containing "name"
        for column in columns:
            name = str(column.get("name", "")).lower()
            if "name" in name:
                return column["name"]

        # Fall back to text columns
        return self._pick_text_column(metadata, table)

    def _pick_temporal_like_column(self, metadata: DatabaseMetadata, table: str) -> str | None:
        table_info = metadata.schema["tables"].get(table, {})
        columns = table_info.get("columns", [])

        for column in columns:
            name = str(column.get("name", "")).lower()
            if name == "year" or name.endswith("_year"):
                return column["name"]

        for column in columns:
            name = str(column.get("name", "")).lower()
            if any(token in name for token in ["date", "time", "year", "semester", "term", "enroll"]):
                return column["name"]

        return None

    def _pick_question_aligned_column(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        table: str,
        *,
        prefer_temporal: bool,
        avoid_identifier: bool,
    ) -> str | None:
        table_info = metadata.schema.get("tables", {}).get(table, {})
        columns = [str(col.get("name", "")) for col in table_info.get("columns", [])]
        if not columns:
            return None

        best_col: str | None = None
        best_score = 0.0
        for col in columns:
            lower_col = col.lower()
            if avoid_identifier and (lower_col == "id" or lower_col.endswith("_id")):
                continue

            score = self._score_column_relevance(lowered_question, lower_col)

            if prefer_temporal and any(token in lower_col for token in ["year", "date", "time", "semester", "term", "enroll"]):
                score += 2.0

            if any(token in lowered_question for token in ["year", "date", "time", "semester", "term", "enrollment"]):
                if any(token in lower_col for token in ["year", "date", "time", "semester", "term", "enroll"]):
                    score += 2.0

            if any(token in lowered_question for token in ["name", "names"]):
                if "name" in lower_col:
                    score += 1.0

            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def _try_generate_groupby_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
        intent: QueryIntent | None = None,
        role_index: SchemaRoleIndex | None = None,
    ) -> str | None:
        """Generate GROUP BY queries for aggregation patterns."""
        role_index = role_index or self._infer_schema_roles(metadata)

        groupby_indicators = [
            "in each", "per", "for each", "group by", "each", "every", "distribution", "breakdown", "by",
        ]
        has_groupby_indicator = any(indicator in lowered_question for indicator in groupby_indicators)
        if not has_groupby_indicator and intent and intent.needs_group_by:
            has_groupby_indicator = True
        if not has_groupby_indicator and self._contains_any(lowered_question, ["top", "highest", "lowest", "most", "least"]):
            has_groupby_indicator = len(context.relevant_tables) >= 2
        if not has_groupby_indicator and self._contains_any(lowered_question, ["count", "number of", "total number"]):
            has_groupby_indicator = len(self._mentioned_tables(lowered_question, context.relevant_tables)) >= 2
        
        if not has_groupby_indicator:
            return None
        
        agg_func = None

        if agg_func is None:
            agg_keywords = {
                "count": "COUNT(*)",
                "total": "COUNT(*)",
                "number": "COUNT(*)",
                "sum": "SUM",
                "average": "AVG",
                "avg": "AVG",
                "maximum": "MAX",
                "minimum": "MIN",
                "max": "MAX",
                "min": "MIN",
                "highest": "MAX",
                "lowest": "MIN",
                "most": "COUNT(*)",
                "least": "COUNT(*)",
            }
            
            for keyword, func in agg_keywords.items():
                if keyword in lowered_question:
                    agg_func = func
                    break

        if agg_func == "COUNT(*)" and self._contains_any(
            lowered_question,
            ["hours", "worked", "salary", "amount", "budget", "revenue", "cost", "price", "wage"],
        ):
            agg_func = "SUM"
        
        if not agg_func:
            return None

        # Average performance/rating by department (or similar grouping entity).
        if (
            agg_func == "AVG"
            and self._contains_any(lowered_question, ["performance", "rating", "score", "grade"])
            and self._contains_any(lowered_question, ["department", "team", "group"])
        ):
            group_table = self._find_table_by_keywords(metadata, [["department", "departments", "team", "teams", "group", "groups"]])
            entity_table = self._find_table_by_keywords(metadata, [["employee", "employees", "student", "students", "staff", "worker", "workers"]])
            metric_table = self._find_table_by_keywords(metadata, [["performance", "rating", "score", "grade", "review"]])

            if group_table and entity_table and metric_table:
                group_id = self._find_table_id_column(metadata, group_table)
                group_label = self._pick_name_like_column(metadata, group_table) or group_id
                entity_id = self._find_table_id_column(metadata, entity_table)
                fk_entity_to_group = self._find_fk_to_table(metadata, entity_table, group_table)
                fk_metric_to_entity = self._find_fk_to_table(metadata, metric_table, entity_table)

                metric_col = self._find_column_by_keywords(metadata, metric_table, ["rating", "score", "grade"])
                if not metric_col:
                    metric_col = self._pick_numeric_column(metadata, metric_table, lowered_question)

                if all([group_id, group_label, entity_id, fk_entity_to_group, fk_metric_to_entity, metric_col]):
                    return (
                        f"SELECT g.{group_label}, AVG(m.{metric_col}) AS avg_value "
                        f"FROM {group_table} g "
                        f"LEFT JOIN {entity_table} e ON e.{fk_entity_to_group} = g.{group_id} "
                        f"LEFT JOIN {metric_table} m ON m.{fk_metric_to_entity} = e.{entity_id} "
                        f"GROUP BY g.{group_id}, g.{group_label}"
                    )

        # Total hours worked by each employee across projects (or analogous many-to-many bridge metrics).
        if agg_func == "SUM" and self._contains_any(lowered_question, ["hours", "worked"]):
            entity_table = self._find_table_by_keywords(metadata, [["employee", "employees", "staff", "worker", "workers", "student", "students"]])
            metric_table = self._find_table_by_keywords(metadata, [["employee_projects", "assignment", "worklog", "timesheet", "enrollment", "link", "bridge"]])
            if not metric_table and context.relevant_tables:
                metric_table = next(
                    (
                        t
                        for t in context.relevant_tables
                        if self._find_column_by_keywords(metadata, t, ["hours_worked", "hours", "worked_hours"])
                    ),
                    None,
                )

            if entity_table and metric_table:
                entity_id = self._find_table_id_column(metadata, entity_table)
                entity_label = self._pick_name_like_column(metadata, entity_table) or entity_id
                fk_metric_to_entity = self._find_fk_to_table(metadata, metric_table, entity_table)
                metric_col = self._find_column_by_keywords(metadata, metric_table, ["hours_worked", "hours", "worked_hours"])
                if not metric_col:
                    metric_col = self._pick_numeric_column(metadata, metric_table, lowered_question)

                if all([entity_id, entity_label, fk_metric_to_entity, metric_col]):
                    return (
                        f"SELECT e.{entity_label}, SUM(m.{metric_col}) AS total_hours "
                        f"FROM {entity_table} e LEFT JOIN {metric_table} m ON m.{fk_metric_to_entity} = e.{entity_id} "
                        f"GROUP BY e.{entity_id}, e.{entity_label}"
                    )

        primary_table = self._primary_table_from_question(lowered_question, metadata, context)
        group_match = re.search(r"(?:in\s+each|for\s+each|each|per)\s+([a-z_]+)", lowered_question)
        group_table = None
        if group_match:
            group_candidates = self._candidate_tables_from_phrase(metadata, group_match.group(1))
            if group_candidates:
                group_table = group_candidates[0]

        if not group_table and primary_table and self._contains_any(lowered_question, ["number of", "count", "average", "maximum", "minimum", "highest", "lowest"]):
            group_table = primary_table

        # Count related entities (e.g., number of students per course/department).
        if agg_func == "COUNT(*)" and group_table:
            mentioned = self._mentioned_tables(lowered_question, context.relevant_tables)
            detail_table = next((t for t in mentioned if t != group_table), None)
            if not detail_table:
                # Fall back to any child table that clearly references the group table.
                child_candidates = context.relevant_tables or list(metadata.schema.get("tables", {}).keys())
                detail_table = next(
                    (
                        candidate
                        for candidate in child_candidates
                        if candidate != group_table and self._find_fk_to_table(metadata, candidate, group_table)
                    ),
                    None,
                )
            if detail_table:
                # Direct child->group relationship.
                fk_detail_to_group = self._find_fk_to_table(metadata, detail_table, group_table)
                group_id = self._find_table_id_column(metadata, group_table)
                group_label = self._pick_name_like_column(metadata, group_table) or group_id
                detail_id = self._find_table_id_column(metadata, detail_table)
                if fk_detail_to_group and group_id and group_label and detail_id:
                    limit_sql = " LIMIT 1" if any(term in lowered_question for term in ["highest", "most", "lowest", "least"]) else ""
                    order_sql = " ORDER BY total_count DESC" if any(term in lowered_question for term in ["highest", "most"]) else ""
                    if any(term in lowered_question for term in ["lowest", "least"]):
                        order_sql = " ORDER BY total_count ASC"
                    if limit_sql:
                        order_sql_count = f" ORDER BY COUNT(d.{detail_id}) DESC" if "DESC" in order_sql else f" ORDER BY COUNT(d.{detail_id}) ASC"
                        return (
                            f"SELECT g.{group_label} "
                            f"FROM {group_table} g JOIN {detail_table} d ON d.{fk_detail_to_group} = g.{group_id} "
                            f"GROUP BY g.{group_id}{order_sql_count} LIMIT 1"
                        )
                    return (
                        f"SELECT g.{group_label}, COUNT(d.{detail_id}) AS total_count "
                        f"FROM {group_table} g LEFT JOIN {detail_table} d ON d.{fk_detail_to_group} = g.{group_id} "
                        f"GROUP BY g.{group_id}, g.{group_label}{order_sql}{limit_sql}"
                    )

                # Many-to-many relationship via bridge table.
                bridge = self._find_bridge_table(metadata, group_table, detail_table)
                if bridge:
                    bridge_table, fk_group, fk_detail = bridge
                    group_id = self._find_table_id_column(metadata, group_table)
                    group_label = self._pick_name_like_column(metadata, group_table) or group_id
                    if group_id and group_label:
                        limit_sql = " LIMIT 1" if any(term in lowered_question for term in ["highest", "most", "lowest", "least"]) else ""
                        order_sql = " ORDER BY total_count DESC" if any(term in lowered_question for term in ["highest", "most"]) else ""
                        if any(term in lowered_question for term in ["lowest", "least"]):
                            order_sql = " ORDER BY total_count ASC"
                        return (
                            f"SELECT g.{group_label}, COUNT(b.{fk_detail}) AS total_count "
                            f"FROM {group_table} g LEFT JOIN {bridge_table} b ON g.{group_id} = b.{fk_group} "
                            f"GROUP BY g.{group_id}, g.{group_label}{order_sql}{limit_sql}"
                        )

        # Aggregated metric by requested group (e.g., average/max grade per course).
        if agg_func in {"AVG", "MAX", "MIN"} and group_table:
            metric_table = self._find_table_by_keywords(metadata, [["grade", "score", "rating", "mark"]])
            if not metric_table and context.relevant_tables:
                metric_table = next(
                    (t for t in context.relevant_tables if t != group_table and self._pick_numeric_column(metadata, t, lowered_question)),
                    None,
                )
            metric_col = self._pick_numeric_column(metadata, metric_table, lowered_question) if metric_table else None
            if metric_table and metric_col:
                fk_metric_to_group = self._find_fk_to_table(metadata, metric_table, group_table)
                group_id = self._find_table_id_column(metadata, group_table)
                group_label = self._pick_name_like_column(metadata, group_table) or group_id
                if fk_metric_to_group and group_id and group_label:
                    alias_name = "avg_value" if agg_func == "AVG" else "value"
                    return (
                        f"SELECT g.{group_label}, {agg_func}(m.{metric_col}) AS {alias_name} "
                        f"FROM {group_table} g LEFT JOIN {metric_table} m ON g.{group_id} = m.{fk_metric_to_group} "
                        f"GROUP BY g.{group_id}, g.{group_label}"
                    )

        # Generic "count of X by each Y" without forcing a join when child has parent FK.
        count_child_match = re.search(r"(?:number of|count of|count)\s+([a-z_]+)", lowered_question)
        count_parent_match = re.search(r"each\s+([a-z_]+)", lowered_question)
        if agg_func == "COUNT(*)" and count_child_match and count_parent_match:
            child_phrase = count_child_match.group(1)
            parent_phrase = count_parent_match.group(1)

            child_candidates = self._candidate_tables_from_phrase(metadata, child_phrase)
            parent_candidates = self._candidate_tables_from_phrase(metadata, parent_phrase)

            if context.relevant_tables:
                preferred_child = [t for t in child_candidates if t in context.relevant_tables]
                if preferred_child:
                    child_candidates = preferred_child + [t for t in child_candidates if t not in preferred_child]

            child_table = child_candidates[0] if child_candidates else None
            parent_table = parent_candidates[0] if parent_candidates else None
            if child_table and parent_table:
                child_cols = [
                    str(col.get("name", ""))
                    for col in metadata.schema.get("tables", {}).get(child_table, {}).get("columns", [])
                ]
                parent_token = parent_phrase.lower()
                parent_singular = parent_token[:-1] if parent_token.endswith("s") else parent_token

                fk_candidates = [
                    col for col in child_cols
                    if col.lower().endswith("id") and (parent_token in col.lower() or parent_singular in col.lower())
                ]
                if not fk_candidates:
                    inferred_fk = self._find_fk_to_table(metadata, child_table, parent_table)
                    if inferred_fk:
                        fk_candidates = [inferred_fk]

                if fk_candidates:
                    group_fk = fk_candidates[0]
                    return (
                        f"SELECT {group_fk}, COUNT(*) AS total_count "
                        f"FROM {child_table} "
                        f"GROUP BY {group_fk}"
                    )
        
        table = self._choose_best_table(
            lowered_question,
            metadata,
            context,
            prioritize_metric_table=True,
            intent=intent,
            role_index=role_index,
        )
        if not table:
            return None
        
        # When we have multiple relevant tables, prioritize the one mentioned in the question
        if context.relevant_tables and len(context.relevant_tables) > 1:
            for candidate_table in context.relevant_tables:
                if candidate_table in lowered_question:
                    table = candidate_table
                    break
        if not table:
            return None
        
        groupby_col = None
        table_info = metadata.schema["tables"].get(table, {})
        columns = table_info.get("columns", [])

        # Prefer label-like columns for grouping.
        label_candidates = role_index.label_columns.get(table, [])
        if label_candidates:
            groupby_col = label_candidates[0]
        
        if not groupby_col:
            # Use first non-ID, non-foreign key column
            for column in columns:
                col_name = str(column.get("name", "")).lower()
                if "id" not in col_name and "_id" not in col_name:
                    groupby_col = column["name"]
                    break
        
        if not groupby_col and columns:
            groupby_col = columns[0]["name"]
        
        if not groupby_col:
            return None

        # Detect ORDER BY modifiers (for "highest", "most", etc.)
        order_by_desc = any(word in lowered_question for word in ["highest", "maximum", "most", "best"])
        order_clause = " ORDER BY " + (f"COUNT(*)" if "COUNT" in agg_func else "value") + " DESC" if order_by_desc else ""
        limit_clause = " LIMIT 1" if order_by_desc else ""

        # Generic "top entities by summed metric" pattern (e.g., spent/sold totals).
        if "SUM" in agg_func and context.relevant_tables and len(context.relevant_tables) >= 2:
            entity_table = None
            for candidate in context.relevant_tables:
                if self._table_mentioned_in_question(lowered_question, candidate):
                    entity_table = candidate
                    break

            if entity_table:
                metric_table = None
                metric_col = None
                best_metric_score = -1.0
                for candidate in context.relevant_tables:
                    if candidate == entity_table:
                        continue
                    candidate_metric = self._pick_numeric_column(metadata, candidate, lowered_question)
                    if not candidate_metric:
                        continue

                    col_lower = candidate_metric.lower()
                    score = self._score_column_relevance(lowered_question, col_lower)
                    if ("total" in lowered_question or "totals" in lowered_question) and "total" in col_lower:
                        score += 3.0
                    if "quantity" in lowered_question and "quantity" in col_lower:
                        score += 3.0
                    if "amount" in lowered_question and "amount" in col_lower:
                        score += 3.0
                    if self._table_mentioned_in_question(lowered_question, candidate):
                        score += 1.0

                    if score > best_metric_score:
                        best_metric_score = score
                        metric_table = candidate
                        metric_col = candidate_metric

                if metric_table and metric_col:
                    join_info = self._find_join_pair(metadata, [metric_table, entity_table])
                    if join_info:
                        child_table, parent_table, child_col, parent_col = join_info
                        entity_label = self._pick_name_like_column(metadata, entity_table) or self._find_table_id_column(metadata, entity_table)
                        entity_id = self._find_table_id_column(metadata, entity_table)
                        if entity_label:
                            limit_n = self._extract_top_n(lowered_question, default_value=1)
                            limit_sql = f" LIMIT {limit_n}" if order_by_desc else ""

                            if child_table == metric_table and parent_table == entity_table:
                                group_cols = [f"e.{entity_label}"]
                                if entity_id and entity_id != entity_label:
                                    group_cols.insert(0, f"e.{entity_id}")
                                return (
                                    f"SELECT e.{entity_label}, SUM(m.{metric_col}) AS total "
                                    f"FROM {entity_table} e JOIN {metric_table} m ON m.{child_col} = e.{parent_col} "
                                    f"GROUP BY {', '.join(group_cols)} ORDER BY total DESC{limit_sql}"
                                )
        
        # Check for JOIN case (e.g., "average age of students in each department")
        if context.relevant_tables and len(context.relevant_tables) >= 2:
            # Try to find a bridge table for many-to-many relationships first
            bridge = self._find_bridge_table(metadata, context.relevant_tables[0], context.relevant_tables[1])
            if bridge and "COUNT" in str(agg_func):
                bridge_table, fk_t1, fk_t2 = bridge
                t1, t2 = context.relevant_tables[0], context.relevant_tables[1]
                t1_id = self._find_table_id_column(metadata, t1)
                t1_name = self._pick_name_like_column(metadata, t1)
                if t1_id and t1_name:
                    order_clause_mm = " ORDER BY COUNT(*) DESC" if order_by_desc else ""
                    final_limit = f" LIMIT {self._extract_top_n(lowered_question, default_value=1)}" if order_by_desc else ""
                    return (
                        f"SELECT t1.{t1_name}, COUNT(x.{fk_t2}) AS total_count "
                        f"FROM {t1} t1 LEFT JOIN {bridge_table} x ON t1.{t1_id} = x.{fk_t1} "
                        f"GROUP BY t1.{t1_id}, t1.{t1_name}{order_clause_mm}{final_limit}"
                    )
            
            # Check for regular 1:N JOIN case (e.g., "average age of students in each department")
            join_info = self._find_join_pair(metadata, context.relevant_tables)
            if join_info:
                child_table, parent_table, child_col, parent_col = join_info
                parent_label_col = self._pick_name_like_column(metadata, parent_table) or parent_col
                child_metric_col = self._pick_numeric_column(metadata, child_table, lowered_question)

                if "AVG" in agg_func and child_metric_col:
                    return (
                        f"SELECT p.{parent_label_col}, "
                        f"AVG(c.{child_metric_col}) AS avg_value "
                        f"FROM {child_table} c JOIN {parent_table} p ON c.{child_col} = p.{parent_col} "
                        f"GROUP BY p.{parent_label_col}"
                    )

                if "COUNT" in agg_func:
                    child_pk = self._find_column_by_keywords(metadata, child_table, ["id", f"{child_table[:-1]}_id"]) or child_col
                    join_type = "LEFT JOIN" if self._contains_any(lowered_question, ["total number", "number of", "count"]) else "JOIN"
                    order_clause_count = " ORDER BY COUNT(*) DESC" if order_by_desc else ""
                    return (
                        f"SELECT p.{parent_label_col}, COUNT(c.{child_pk}) AS total_count "
                        f"FROM {parent_table} p {join_type} {child_table} c ON c.{child_col} = p.{parent_col} "
                        f"GROUP BY p.{parent_label_col}{order_clause_count}{limit_clause}"
                    )

                if "SUM" in agg_func and child_metric_col:
                    order_clause_sum = " ORDER BY SUM(" + child_metric_col + ") DESC" if order_by_desc else ""
                    # Extract limit from "top N" pattern
                    limit_n = self._extract_top_n(lowered_question, default_value=1)
                    final_limit = f" LIMIT {limit_n}" if order_by_desc else ""
                    return (
                        f"SELECT p.{parent_label_col}, SUM(c.{child_metric_col}) AS total "
                        f"FROM {child_table} c JOIN {parent_table} p ON c.{child_col} = p.{parent_col} "
                        f"GROUP BY p.{parent_col}, p.{parent_label_col}{order_clause_sum}{final_limit}"
                    )

                if ("MAX" in agg_func or "MIN" in agg_func) and child_metric_col:
                    agg_type = "MAX" if "MAX" in agg_func else "MIN"
                    order_clause_mm = f" ORDER BY {agg_type}(c.{child_metric_col}) DESC" if order_by_desc else ""
                    return (
                        f"SELECT p.{parent_label_col}, {agg_type}(c.{child_metric_col}) AS value "
                        f"FROM {child_table} c JOIN {parent_table} p ON c.{child_col} = p.{parent_col} "
                        f"GROUP BY p.{parent_label_col}{order_clause_mm}{limit_clause}"
                    )
        
        # Simple GROUP BY
        if "SUM" in agg_func or "COUNT" in agg_func:
            metric_col = None
            if "SUM" in agg_func or ("sum" not in lowered_question and "count" not in lowered_question):
                metric_col = self._pick_numeric_column(metadata, table, lowered_question)
            
            if metric_col and "SUM" in agg_func:
                return f"SELECT {groupby_col}, SUM({metric_col}) AS total FROM {table} GROUP BY {groupby_col}{order_clause}{limit_clause}"
            elif "COUNT" in agg_func or "count" in lowered_question or "total" in lowered_question:
                # For "most" pattern with COUNT, order by count DESC
                order_clause_count = " ORDER BY COUNT(*) DESC" if order_by_desc else ""
                return f"SELECT {groupby_col}, COUNT(*) AS count FROM {table} GROUP BY {groupby_col}{order_clause_count}{limit_clause}"
        
        if "AVG" in agg_func:
            metric_col = self._pick_numeric_column(metadata, table, lowered_question)
            if metric_col:
                return f"SELECT {groupby_col}, AVG({metric_col}) AS average FROM {table} GROUP BY {groupby_col}{order_clause}{limit_clause}"
        
        if "MAX" in agg_func or "MIN" in agg_func:
            metric_col = self._pick_numeric_column(metadata, table, lowered_question)
            if metric_col:
                agg_type = "MAX" if "MAX" in agg_func else "MIN"
                order_clause_mm = f" ORDER BY {agg_type}({metric_col}) DESC" if order_by_desc else ""
                return f"SELECT {groupby_col}, {agg_type}({metric_col}) AS value FROM {table} GROUP BY {groupby_col}{order_clause_mm}{limit_clause}"
        
        return None

    def _try_generate_where_clause(self, lowered_question: str, metadata: DatabaseMetadata, table: str) -> str | None:
        """Generate WHERE clause conditions."""
        if not table:
            return None
        
        table_info = metadata.schema["tables"].get(table, {})
        columns = table_info.get("columns", [])

        # Generic unassigned/null owner filters (e.g., courses not assigned to any instructor).
        if self._contains_any(lowered_question, ["not assigned", "without", "unassigned", "not allocated", "no "]) and self._contains_any(lowered_question, ["instructor", "advisor", "teacher", "manager", "owner"]):
            for col in columns:
                col_name = str(col.get("name", ""))
                col_lower = col_name.lower()
                if col_lower.endswith("_id") and any(token in col_lower for token in ["instructor", "advisor", "teacher", "manager", "owner"]):
                    return f"{col_name} IS NULL"

        # Generic active/current filters.
        if self._contains_any(lowered_question, ["active", "currently active", "current"]):
            status_candidates: list[str] = []
            temporal_candidates: list[str] = []
            for col in columns:
                col_name = str(col.get("name", ""))
                col_lower = col_name.lower()
                if any(token in col_lower for token in ["status", "state", "active", "enabled", "flag"]):
                    status_candidates.append(col_name)
                if any(token in col_lower for token in ["semester", "term", "current", "start", "end", "date", "year"]):
                    temporal_candidates.append(col_name)

            if status_candidates:
                preferred_status = next((c for c in status_candidates if any(tok in c.lower() for tok in ["status", "state"])), None)
                status_col = preferred_status or status_candidates[0]
                if any(token in status_col.lower() for token in ["status", "state"]):
                    return f"LOWER({status_col}) IN ('active', 'current')"
                if any(token in status_col.lower() for token in ["active", "enabled", "flag"]):
                    return f"{status_col} = 1"
                return f"LOWER({status_col}) IN ('active', 'current')"

            if temporal_candidates and self._contains_any(lowered_question, ["semester", "term"]):
                temporal_col = temporal_candidates[0]
                if "current" in lowered_question:
                    return f"{temporal_col} = 'current'"
                return f"{temporal_col} IS NOT NULL"

        specific_entity_match = re.search(r"\bspecific\s+([a-z_]+)\b", lowered_question)
        if specific_entity_match:
            entity_token = specific_entity_match.group(1)
            for col in columns:
                col_name = str(col.get("name", ""))
                col_lower = col_name.lower()
                if col_lower.endswith("_id") and entity_token in col_lower:
                    return f"{col_name} = ?"
        
        # Age comparison patterns - handle "older than 20" and "older 20"
        age_match = re.search(r'(older|younger|at least|more than|less than|greater than|above|below)\s+(?:than\s+)?(\d+)', lowered_question)
        if age_match:
            op, value = age_match.groups()
            age_col = None
            for col in columns:
                if "age" in str(col.get("name", "")).lower():
                    age_col = col["name"]
                    break
            
            if age_col:
                if op in ["older", "at least", "more than", "greater than", "above"]:
                    return f"{age_col} > {value}"
                elif op in ["younger", "less than", "below"]:
                    return f"{age_col} < {value}"
        
        # Year comparison patterns
        year_match = re.search(r'(after|before|in|from|since)\s+(\d{4})', lowered_question)
        if year_match:
            op, year = year_match.groups()
            year_col = None
            for col in columns:
                col_name = str(col.get("name", "")).lower()
                if "year" in col_name or "date" in col_name:
                    year_col = col["name"]
                    break
            
            if year_col:
                if op in ["after", "from", "since"]:
                    if op == "after":
                        return f"{year_col} > {year}"
                    return f"{year_col} >= {year}"
                elif op in ["before"]:
                    return f"{year_col} < {year}"

        # Generic country/location filters like "customers from brazil".
        from_match = re.search(r"\bfrom\s+([a-z][a-z\s]+?)\b", lowered_question)
        if from_match:
            location = from_match.group(1).strip()
            location_col = None
            for col in columns:
                col_name = str(col.get("name", "")).lower()
                if any(token in col_name for token in ["country", "city", "state", "region", "location"]):
                    location_col = col["name"]
                    break
            if location_col:
                return f"{location_col} = '{location.title()}'"

        numeric_cmp = re.search(
            r"(more than|greater than|less than|at least|at most|above|below)\s+(\d+)",
            lowered_question,
        )
        if numeric_cmp:
            phrase, value = numeric_cmp.groups()
            op_map = {
                "more than": ">",
                "greater than": ">",
                "above": ">",
                "less than": "<",
                "below": "<",
                "at least": ">=",
                "at most": "<=",
            }
            op = op_map.get(phrase)
            if op:
                candidates: list[tuple[float, str]] = []
                for col in columns:
                    col_name = str(col.get("name", ""))
                    col_lower = col_name.lower()
                    col_type = str(col.get("type", "")).lower()
                    if "id" in col_lower:
                        continue
                    if not any(token in col_type for token in ["int", "real", "num", "decimal", "float"]):
                        continue
                    score = self._score_column_relevance(lowered_question, col_lower)
                    if col_lower in lowered_question:
                        score += 2.0
                    candidates.append((score, col_name))

                if candidates:
                    candidates.sort(reverse=True)
                    return f"{candidates[0][1]} {op} {value}"
        
        return None

    def _try_generate_simple_select_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        """Try to generate a simple single-table SELECT for non-aggregation queries."""
        
        # Only trigger on "list", "all", "get" patterns (or context-guided)
        if not any(term in lowered_question for term in ["list", "all", "get", "show", "retrieve"]):
            # Unless we have very strong context signals
            if not context.relevant_columns.get(context.relevant_tables[0] if context.relevant_tables else None):
                return None
        
        # Skip if this is a multi-table join question (e.g., "along with")
        if any(term in lowered_question for term in ["along with", "together with", "with their", "with the"]):
            return None

        if len(self._mentioned_tables(lowered_question, context.relevant_tables)) >= 2:
            return None
        
        # Skip if question requires aggregation or grouping (use word boundaries)
        if any(re.search(rf'\b{term}\b', lowered_question) for term in ["count", "sum", "average", "avg", "max", "min", "total", "group", "top", "highest", "most", "least", "lowest"]):
            return None
        
        # Skip if question asks for filters or comparisons
        if any(term in lowered_question for term in ["where", "filter", "more than", "less than", "greater than"]):
            return None
        
        # Get the best table for this question
        table = self._primary_table_from_question(lowered_question, metadata, context) or self._choose_best_table(
            lowered_question,
            metadata,
            context,
        )
        if not table:
            return None

        if self._contains_any(lowered_question, ["names of", "name of"]):
            name_col = self._pick_name_like_column(metadata, table)
            if name_col:
                return f"SELECT {name_col} FROM {table}"

        if self._contains_any(lowered_question, ["all", "list all", "show all", "retrieve all", "display all"]):
            return f"SELECT * FROM {table}"
        
        # Use context-provided relevant columns, as these are embedding-scored
        if context.relevant_columns.get(table):
            relevant_cols = context.relevant_columns[table]
            if relevant_cols:
                # Add other columns that appear in the question to improve completeness
                table_info = metadata.schema["tables"].get(table, {})
                all_cols = [col["name"] for col in table_info.get("columns", [])]
                for col in all_cols:
                    if col not in relevant_cols and "name" in col.lower() and len(relevant_cols) < 10:
                        relevant_cols = list(relevant_cols) + [col]
                
                return f"SELECT {', '.join(relevant_cols[:10])} FROM {table}"
        
        # Fall back: select all non-ID columns
        table_info = metadata.schema["tables"].get(table, {})
        all_columns = [col["name"] for col in table_info.get("columns", [])]
        relevant_cols = [col for col in all_columns if "id" not in col.lower()][:10]
        
        if relevant_cols:
            return f"SELECT {', '.join(relevant_cols)} FROM {table}"
        
        return None

    def _try_generate_lookup_by_id_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        if self._contains_any(
            lowered_question,
            ["more than", "less than", "greater than", "at least", "at most", "top", "highest", "lowest"],
        ):
            return None

        match = re.search(
            r"\b(?:details of|detail of|show|list)\s+([a-z_ ]+?)\s+(?:with\s+)?id\s*(?:=|is)?\s*(\d+)\b",
            lowered_question,
        )
        if not match:
            return None

        entity_phrase = match.group(1).strip()
        numeric_id = match.group(2)
        candidates = self._candidate_tables_from_phrase(metadata, entity_phrase)
        if not candidates:
            candidates = context.relevant_tables or list(metadata.schema.get("tables", {}).keys())

        for table in candidates:
            id_col = self._find_table_id_column(metadata, table)
            if id_col:
                columns = [str(col.get("name", "")) for col in metadata.schema.get("tables", {}).get(table, {}).get("columns", [])]
                select_cols = columns[:5] if columns else ["*"]
                return f"SELECT {', '.join(select_cols)} FROM {table} WHERE {id_col} = {numeric_id}"
        return None

    def _try_generate_order_by_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
        table: str,
        intent: QueryIntent | None = None,
        role_index: SchemaRoleIndex | None = None,
    ) -> str | None:
        """Generate ORDER BY queries for top/ranking patterns."""
        role_index = role_index or self._infer_schema_roles(metadata)
        order_indicators = ["top", "highest", "lowest", "oldest", "youngest", "most", "least", "first", "last"]
        has_order_indicator = any(indicator in lowered_question for indicator in order_indicators)
        
        if not has_order_indicator or not table:
            return None
        
        metric_col = self._pick_numeric_column(metadata, table, lowered_question) or "*"
        if metric_col == "*" and role_index.metric_columns.get(table):
            metric_col = role_index.metric_columns[table][0]
        label_col = self._pick_text_column(metadata, table) or "*"
        
        # Determine sort direction
        is_ascending = any(word in lowered_question for word in ["lowest", "youngest", "least", "first"])
        sort_direction = "ASC" if is_ascending else "DESC"
        
        # Determine limit
        limit = intent.top_n if intent and intent.top_n else 10
        limit_match = re.search(r'top\s+(\d+)', lowered_question)
        if limit_match:
            limit = int(limit_match.group(1))
        
        if metric_col != "*":
            return f"SELECT {label_col}, {metric_col} FROM {table} ORDER BY {metric_col} {sort_direction} LIMIT {limit}"
        else:
            return f"SELECT * FROM {table} ORDER BY {label_col} {sort_direction} LIMIT {limit}"

    def _try_generate_above_average_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:



        """Generate queries for above-average patterns."""
        if not any(
            phrase in lowered_question
            for phrase in [
                "above average",
                "above the average",
                "greater than average",
                "more than the average",
                "more than average",
                "than the average",
            ]
        ):
            return None

        target_table = self._primary_table_from_question(lowered_question, metadata, context)
        metric_table = None
        metric_col = None
        best_metric_score = -1.0

        for table_name, table_info in metadata.schema.get("tables", {}).items():
            for col in table_info.get("columns", []):
                col_name = str(col.get("name", ""))
                col_lower = col_name.lower()
                col_type = str(col.get("type", "")).lower()
                if "id" in col_lower:
                    continue
                if not any(token in col_type for token in ["int", "real", "num", "decimal", "float"]):
                    continue

                score = self._score_column_relevance(lowered_question, col_lower)
                for token in ["salary", "rating", "grade", "score", "hours", "amount", "budget", "revenue", "cost", "price", "wage"]:
                    if token in lowered_question and token in col_lower:
                        score += 4.0
                if table_name.lower() in lowered_question:
                    score += 1.0

                if score > best_metric_score:
                    best_metric_score = score
                    metric_table = table_name
                    metric_col = col_name

        if not metric_table:
            metric_table = self._find_table_by_keywords(metadata, [["grade", "score", "rating", "mark", "salary", "amount", "hours"]])
        if metric_table and not metric_col:
            metric_col = self._pick_numeric_column(metadata, metric_table, lowered_question)
        if not metric_table:
            metric_table = self._choose_best_table(lowered_question, metadata, context, prioritize_metric_table=True)
            metric_col = self._pick_numeric_column(metadata, metric_table, lowered_question) if metric_table else None

        if not metric_col:
            return None

        if target_table and target_table != metric_table:
            target_id = self._find_table_id_column(metadata, target_table)
            target_label = self._pick_name_like_column(metadata, target_table) or target_id
            fk_metric_to_target = self._find_fk_to_table(metadata, metric_table, target_table)
            if target_id and target_label and fk_metric_to_target:
                return (
                    f"SELECT DISTINCT t.{target_label} "
                    f"FROM {target_table} t JOIN {metric_table} m ON t.{target_id} = m.{fk_metric_to_target} "
                    f"WHERE m.{metric_col} > (SELECT AVG({metric_col}) FROM {metric_table})"
                )

        label_col = self._pick_name_like_column(metadata, metric_table)
        projection = f"{label_col}, {metric_col}" if label_col else metric_col
        return (
            f"SELECT {projection} FROM {metric_table} "
            f"WHERE {metric_col} > (SELECT AVG({metric_col}) FROM {metric_table})"
        )

    def _try_generate_avg_metric_by_group_sql(self, lowered_question: str, metadata: DatabaseMetadata) -> str | None:
        if not self._contains_any(lowered_question, ["average", "avg"]):
            return None
        if not self._contains_any(lowered_question, ["performance", "rating", "score", "grade"]):
            return None
        if not self._contains_any(lowered_question, ["each", "per", "by"]):
            return None
        if not self._contains_any(lowered_question, ["department", "team", "group"]):
            return None

        group_table = self._find_table_by_keywords(metadata, [["department", "departments", "team", "teams", "group", "groups"]])
        entity_table = self._find_table_by_keywords(metadata, [["employee", "employees", "student", "students", "staff", "worker", "workers"]])
        metric_table = self._find_table_by_keywords(metadata, [["performance", "rating", "score", "grade", "review"]])

        if not (group_table and entity_table and metric_table):
            return None

        group_id = self._find_table_id_column(metadata, group_table)
        group_label = self._pick_name_like_column(metadata, group_table) or group_id
        entity_id = self._find_table_id_column(metadata, entity_table)
        fk_entity_to_group = self._find_fk_to_table(metadata, entity_table, group_table)
        fk_metric_to_entity = self._find_fk_to_table(metadata, metric_table, entity_table)

        metric_col = self._find_column_by_keywords(metadata, metric_table, ["rating", "score", "grade"])
        if not metric_col:
            metric_col = self._pick_numeric_column(metadata, metric_table, lowered_question)

        if not all([group_id, group_label, entity_id, fk_entity_to_group, fk_metric_to_entity, metric_col]):
            return None

        select_group_col = f"g.{group_label}"
        group_by_cols = [f"g.{group_id}"]
        if group_label != group_id:
            group_by_cols.append(select_group_col)

        return (
            f"SELECT {select_group_col}, AVG(m.{metric_col}) AS average_grade "
            f"FROM {group_table} g "
            f"LEFT JOIN {entity_table} e ON e.{fk_entity_to_group} = g.{group_id} "
            f"LEFT JOIN {metric_table} m ON m.{fk_metric_to_entity} = e.{entity_id} "
            f"GROUP BY {', '.join(group_by_cols)}"
        )

    def _try_generate_total_hours_by_entity_sql(self, lowered_question: str, metadata: DatabaseMetadata) -> str | None:
        if not self._contains_any(lowered_question, ["hours", "worked"]):
            return None
        if not self._contains_any(lowered_question, ["total", "sum"]):
            return None
        if not self._contains_any(lowered_question, ["each", "per", "by"]):
            return None
        if not self._contains_any(lowered_question, ["employee", "employees", "staff", "worker", "workers", "student", "students"]):
            return None

        entity_table = self._find_table_by_keywords(metadata, [["employee", "employees", "staff", "worker", "workers", "student", "students"]])
        if not entity_table:
            return None

        metric_table = None
        metric_col = None
        for table_name, table_info in metadata.schema.get("tables", {}).items():
            for col in table_info.get("columns", []):
                col_name = str(col.get("name", ""))
                col_lower = col_name.lower()
                if any(token in col_lower for token in ["hours_worked", "worked_hours", "work_hours", "hours"]):
                    metric_table = table_name
                    metric_col = col_name
                    break
            if metric_table:
                break

        if not (metric_table and metric_col):
            return None

        entity_id = self._find_table_id_column(metadata, entity_table)
        entity_label = self._pick_name_like_column(metadata, entity_table) or entity_id
        fk_metric_to_entity = self._find_fk_to_table(metadata, metric_table, entity_table)

        if not all([entity_id, entity_label, fk_metric_to_entity]):
            return None

        return (
            f"SELECT e.{entity_label}, SUM(m.{metric_col}) AS total_hours "
            f"FROM {entity_table} e LEFT JOIN {metric_table} m ON m.{fk_metric_to_entity} = e.{entity_id} "
            f"GROUP BY e.{entity_id}, e.{entity_label}"
        )

    def _try_generate_at_least_one_sql(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        if "at least one" not in lowered_question:
            return None

        join_info = self._find_join_pair(metadata, context.relevant_tables)
        if not join_info:
            return None

        child_table, parent_table, child_col, parent_col = join_info
        parent_label_col = self._pick_name_like_column(metadata, parent_table) or parent_col
        return (
            f"SELECT DISTINCT p.{parent_label_col} AS {parent_table}_name "
            f"FROM {parent_table} p "
            f"JOIN {child_table} c ON c.{child_col} = p.{parent_col}"
        )
    def _score_table(self, lowered_question: str, table: str, metadata: DatabaseMetadata) -> int:
        score = 0
        tokens = self._tokens(lowered_question)
        table_tokens = self._tokens(table)
        if table in lowered_question:
            score += 5
        score += len(tokens & table_tokens) * 3

        if self._has_entity_id_phrase(lowered_question, table):
            score += 10

        for column in metadata.schema["tables"].get(table, {}).get("columns", []):
            col_name = str(column.get("name", "")).lower()
            col_tokens = self._tokens(col_name)
            score += len(tokens & col_tokens) * 2

        return score

    def _score_column(self, lowered_question: str, table: str, column: str, metadata: DatabaseMetadata) -> int:
        score = 0
        tokens = self._tokens(lowered_question)
        column_tokens = self._tokens(column)
        if column in lowered_question:
            score += 5
        score += len(tokens & column_tokens) * 4

        if any(word in lowered_question for word in ["name", "names"]) and "name" in column.lower():
            score += 2

        return score

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z_]+", text.lower()))

    def _contains_any(self, text: str, terms: list[str]) -> bool:
        lowered = text.lower()
        tokens = self._tokens(lowered)
        for term in terms:
            term_lower = term.lower()
            if " " in term_lower:
                if term_lower in lowered:
                    return True
                continue

            singular = term_lower[:-1] if term_lower.endswith("s") else term_lower
            plural = term_lower if term_lower.endswith("s") else f"{term_lower}s"

            if term_lower in tokens or singular in tokens or plural in tokens:
                return True

            if re.search(rf"\b{re.escape(term_lower)}\b", lowered):
                return True
        return False

    def _table_mentioned_in_question(self, lowered_question: str, table: str) -> bool:
        tokens = self._tokens(lowered_question)
        table_tokens = self._tokens(table)
        if not table_tokens:
            return False
        for token in table_tokens:
            singular = token[:-1] if token.endswith("s") else token
            plural = token if token.endswith("s") else f"{token}s"
            if token in tokens or singular in tokens or plural in tokens:
                return True
        return table.lower() in lowered_question

    def _extract_relation_detail_phrase(self, lowered_question: str) -> str:
        cues = [
            "along with their",
            "along with the",
            "along with",
            "with their",
            "and their",
            "together with their",
            "together with the",
            "together with",
        ]
        for cue in cues:
            if cue in lowered_question:
                tail = lowered_question.split(cue, 1)[1]
                tail = re.split(r"\b(?:who|that|where|for|in)\b", tail, maxsplit=1)[0]
                return tail.strip(" .?;")
        return ""

    def _pick_table_column_from_phrase(self, metadata: DatabaseMetadata, table: str, phrase: str) -> str | None:
        if not table or not phrase:
            return None
        table_info = metadata.schema.get("tables", {}).get(table, {})
        columns = [str(col.get("name", "")) for col in table_info.get("columns", [])]
        if not columns:
            return None

        phrase_tokens = set(self._tokens(phrase))
        best_col: str | None = None
        best_score = 0
        for col in columns:
            col_tokens = set(self._tokens(col.replace("_", " ")))
            if not col_tokens:
                continue
            overlap = len(col_tokens.intersection(phrase_tokens))
            if overlap <= 0:
                continue
            if col.lower().endswith("_id"):
                overlap -= 1
            if overlap > best_score:
                best_score = overlap
                best_col = col
        return best_col

    def _mentioned_tables(self, lowered_question: str, candidate_tables: list[str]) -> list[str]:
        return [
            table for table in (candidate_tables or []) if self._table_mentioned_in_question(lowered_question, table)
        ]

    def _primary_table_from_question(
        self,
        lowered_question: str,
        metadata: DatabaseMetadata,
        context: RetrievalContext,
    ) -> str | None:
        lead_match = re.search(
            r"\b(?:list|show|retrieve|get|find|display|rank|count)\s+([a-z_ ]+?)(?:\s+along\s+with|\s+with\s+their|\s+with\s+the|\s+and\b|\s+who\b|\s+that\b|\s+where\b|\?|$)",
            lowered_question,
        )
        phrase = lead_match.group(1).strip() if lead_match else ""
        if phrase:
            candidates = self._candidate_tables_from_phrase(metadata, phrase)
            if candidates:
                return candidates[0]

        mentioned = self._mentioned_tables(lowered_question, context.relevant_tables)
        if mentioned:
            return mentioned[0]
        return None

    def _find_table_by_keywords(
        self,
        metadata: DatabaseMetadata,
        keyword_groups: list[list[str]],
    ) -> str | None:
        tables = metadata.schema.get("tables", {})
        best_table = None
        best_score = -1
        for table_name, info in tables.items():
            name_lower = table_name.lower()
            columns = [str(col.get("name", "")).lower() for col in info.get("columns", [])]
            score = 0
            for group in keyword_groups:
                group_score = 0
                for kw in group:
                    kw_lower = kw.lower()
                    singular = kw_lower[:-1] if kw_lower.endswith("s") else kw_lower
                    plural = kw_lower if kw_lower.endswith("s") else f"{kw_lower}s"

                    # Strongest signal: table name directly represents the concept.
                    if name_lower in {kw_lower, singular, plural}:
                        group_score = max(group_score, 12)
                    elif kw_lower in name_lower or singular in name_lower:
                        group_score = max(group_score, 8)

                    # Weaker signal: concept appears only in columns.
                    if any(col in {kw_lower, singular, plural, f"{kw_lower}_id", f"{singular}_id"} for col in columns):
                        group_score = max(group_score, 4)
                    elif any(kw_lower in col or singular in col for col in columns):
                        group_score = max(group_score, 2)

                score += group_score
            if score > best_score:
                best_score = score
                best_table = table_name
        return best_table if best_score > 0 else None

    def _find_column_by_keywords(self, metadata: DatabaseMetadata, table: str, keywords: list[str]) -> str | None:
        table_info = metadata.schema.get("tables", {}).get(table, {})
        columns = [str(col.get("name", "")) for col in table_info.get("columns", [])]
        lowered = [col.lower() for col in columns]

        for keyword in keywords:
            for original, lower in zip(columns, lowered):
                if lower == keyword:
                    return original

        for keyword in keywords:
            for original, lower in zip(columns, lowered):
                if keyword in lower:
                    return original

        return None

    def _candidate_tables_from_phrase(self, metadata: DatabaseMetadata, phrase: str) -> list[str]:
        tokens = self._tokens(phrase)
        expanded_tokens = set(tokens)
        for token in tokens:
            if token.endswith("id") and len(token) > 2:
                expanded_tokens.add(token[:-2])
        candidates: list[tuple[int, str]] = []
        for table in metadata.schema.get("tables", {}).keys():
            table_tokens = self._tokens(table)
            score = 0
            for token in expanded_tokens:
                singular = token[:-1] if token.endswith("s") else token
                plural = token if token.endswith("s") else f"{token}s"
                if token in table_tokens or singular in table_tokens or plural in table_tokens:
                    score += 3
                elif any(token in table_token or table_token in token for table_token in table_tokens):
                    score += 1
            if score:
                candidates.append((score, table))
        candidates.sort(reverse=True)
        return [table for _, table in candidates]

    def _has_entity_id_phrase(self, lowered_question: str, table: str) -> bool:
        table_tokens = self._tokens(table)
        if not table_tokens:
            return False
        singular = table[:-1] if table.endswith("s") else table
        phrase_patterns = [
            rf"\b{re.escape(table)}\s*id\b",
            rf"\b{re.escape(singular)}\s*id\b",
        ]
        if any(re.search(pattern, lowered_question) for pattern in phrase_patterns):
            return True
        return any(token in lowered_question for token in table_tokens)

    def _find_table_id_column(self, metadata: DatabaseMetadata, table: str) -> str | None:
        table_info = metadata.schema.get("tables", {}).get(table, {})
        columns = [str(col.get("name", "")) for col in table_info.get("columns", [])]
        lowered = [col.lower() for col in columns]
        singular = table[:-1].lower() if table.lower().endswith("s") else table.lower()

        candidates = [f"{singular}_id", "id", f"{singular}id"]
        for candidate in candidates:
            for original, lower in zip(columns, lowered):
                if lower == candidate:
                    return original

        for original, lower in zip(columns, lowered):
            if lower.endswith("_id"):
                return original
        return None

    def _find_fk_to_table(self, metadata: DatabaseMetadata, source_table: str, target_table: str) -> str | None:
        source_info = metadata.schema.get("tables", {}).get(source_table, {})
        cols = [str(col.get("name", "")) for col in source_info.get("columns", [])]
        lowered = [c.lower() for c in cols]
        target_singular = target_table[:-1].lower() if target_table.lower().endswith("s") else target_table.lower()

        for fk in source_info.get("foreign_keys", []):
            if str(fk.get("references_table", "")) == target_table:
                fk_col = str(fk.get("column", ""))
                if fk_col:
                    return fk_col

        for candidate in [f"{target_table.lower()}_id", f"{target_singular}_id", f"{target_singular}id"]:
            for original, lower in zip(cols, lowered):
                if lower == candidate:
                    return original

        for original, lower in zip(cols, lowered):
            if target_singular in lower and lower.endswith("id"):
                return original

        # If source contains the exact target table ID column name, treat it as FK.
        target_id_col = self._find_table_id_column(metadata, target_table)
        if target_id_col:
            target_id_lower = target_id_col.lower()
            if target_id_lower == "id":
                return None
            for original, lower in zip(cols, lowered):
                if lower == target_id_lower:
                    return original
        return None

    def _find_bridge_table(self, metadata: DatabaseMetadata, table_a: str, table_b: str) -> tuple[str, str, str] | None:
        for bridge_name in metadata.schema.get("tables", {}).keys():
            if bridge_name in {table_a, table_b}:
                continue
            fk_a = self._find_fk_to_table(metadata, bridge_name, table_a)
            fk_b = self._find_fk_to_table(metadata, bridge_name, table_b)
            if fk_a and fk_b and fk_a != fk_b:
                return (bridge_name, fk_a, fk_b)
        return None

    def _score_column_relevance(self, lowered_question: str, column_name: str) -> float:
        """Score column relevance to question using TF-IDF style matching."""
        question_tokens = self._tokens(lowered_question)
        column_tokens = self._tokens(column_name)
        
        if not question_tokens or not column_tokens:
            return 0.0
        
        overlap = question_tokens & column_tokens
        if not overlap:
            return 0.0
        
        # Simple relevance score: number of matching tokens
        return float(len(overlap))

    def _infer_all_relationships(self, metadata: DatabaseMetadata) -> dict[tuple[str, str], tuple[str, str]]:
        """Infer parent-child relationships from FK-like column patterns (generic)."""
        relationships: dict[tuple[str, str], tuple[str, str]] = {}
        tables = metadata.schema.get("tables", {})
        
        for source_table, source_info in tables.items():
            source_cols = [str(col.get("name", "")).lower() for col in source_info.get("columns", [])]
            
            for target_table, target_info in tables.items():
                if source_table == target_table:
                    continue
                
                target_id = self._find_table_id_column(metadata, target_table)
                if not target_id:
                    continue
                
                fk_col = self._find_fk_to_table(metadata, source_table, target_table)
                if fk_col:
                    relationships[(source_table, target_table)] = (fk_col, target_id)
        
        return relationships

    def _extract_top_n(self, lowered_question: str, default_value: int) -> int:
        match = re.search(r"top\s+(\d+)", lowered_question)
        if not match:
            return default_value
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            return default_value
