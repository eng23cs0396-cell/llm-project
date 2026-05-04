from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.services.retrieval_service import RetrievalContext, RetrievalService
from app.services.sql_semantic_guard import SQLSemanticGuard
from app.services.sql_generator import SQLGenerator
from app.services.sql_validator import SQLValidator, ValidationResult
from app.storage import DatabaseMetadata


@dataclass
class RetryOutcome:
    sql: str
    attempts: int
    issues: list[str]
    generation_ms: float
    validation_ms: float
    trace: dict | None = None


class RetryEngine:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        generator: SQLGenerator,
        validator: SQLValidator,
        semantic_guard: SQLSemanticGuard | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generator = generator
        self.validator = validator
        self.semantic_guard = semantic_guard or SQLSemanticGuard()

    def generate_with_retries(
        self,
        question: str,
        metadata: DatabaseMetadata,
        few_shot_examples: list,
        max_attempts: int = 4,
    ) -> RetryOutcome:
        context: RetrievalContext = self.retrieval_service.build_context(
            question, metadata, few_shot_examples
        )

        generation_ms = 0.0
        validation_ms = 0.0
        start_gen = perf_counter()
        sql = self.generator.generate(question, metadata, context)
        generation_ms += (perf_counter() - start_gen) * 1000

        if sql == "INVALID_QUERY":
            return RetryOutcome(
                sql=sql,
                attempts=1,
                issues=["Question is not relevant to database"],
                generation_ms=generation_ms,
                validation_ms=validation_ms,
                trace=getattr(self.generator, "_last_trace", None),
            )

        last_issues: list[str] = []

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                compact_error = "; ".join(last_issues)
                if len(compact_error) > 800:
                    compact_error = compact_error[:800]

                start_gen = perf_counter()
                sql = self.generator.repair(
                    question=question,
                    metadata=metadata,
                    context=context,
                    error_message=compact_error,
                    previous_sql=sql,
                )
                generation_ms += (perf_counter() - start_gen) * 1000

            start_val = perf_counter()
            validation: ValidationResult = self.validator.validate(sql, metadata)
            validation_ms += (perf_counter() - start_val) * 1000

            semantic_issues = self.semantic_guard.check(question, sql, metadata)
            if semantic_issues:
                validation.issues.extend(semantic_issues)
                validation.ok = False

            if validation.ok:
                return RetryOutcome(
                    sql=sql,
                    attempts=attempt,
                    issues=[],
                    generation_ms=generation_ms,
                    validation_ms=validation_ms,
                    trace=getattr(self.generator, "_last_trace", None),
                )

            last_issues = validation.issues

        return RetryOutcome(
            sql=sql,
            attempts=max_attempts,
            issues=last_issues,
            generation_ms=generation_ms,
            validation_ms=validation_ms,
            trace=getattr(self.generator, "_last_trace", None),
        )
