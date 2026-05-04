"""
Main FastAPI application for universal Text-to-SQL system.

Pipeline order: FLAN-T5 (basic queries) -> Mistral (complex queries).
"""

from __future__ import annotations

import time
import uuid
import os
import logging
import threading
import re
from difflib import get_close_matches
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    CompareQueryResponse,
    ConnectDatabaseRequest,
    ConnectDatabaseResponse,
    ModelQueryResult,
    QueryMetrics,
    QueryRequest,
    QueryResponse,
    UploadDatabaseResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.executor import SQLExecutor
from app.services.adapter_setup import setup_adapters
from app.services.retrieval_service import RetrievalService
from app.services.retry_engine import RetryEngine
from app.services.schema_service import SchemaService
from app.services.sql_validator import SQLValidator
from app.services.sql_semantic_guard import SQLSemanticGuard
from app.storage import DatabaseMetadata, STORE

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

# Try to import LLM generator
try:
    from app.services.llm_sql_generator import LLMSQLGenerator
    _llm_import_error: Exception | None = None
except Exception as exc:
    LLMSQLGenerator = None  # type: ignore[assignment]
    _llm_import_error = exc

# Use the validated rule-based generator as fallback.
from app.services.sql_generator import SQLGenerator

_rule_gen = SQLGenerator()

app = FastAPI(title="NL-to-SQL Service", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
STATIC_DIR = BASE_DIR / "app" / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize services
schema_service = SchemaService()
embedding_service = EmbeddingService()
retrieval_service = RetrievalService(embedding_service)
semantic_guard = SQLSemanticGuard()
validator = SQLValidator()
executor = SQLExecutor()

# Model paths
training_dir = BASE_DIR / "training" / "artifacts"
flan_model_dir = training_dir / "flan_t5_sql_lora" / "final"
mistral_model_dir = training_dir / "mistral_7b_sql_lora" / "final"

# NEW PIPELINE ORDER: FLAN-T5 first (optimized for basic queries), then Mistral, then Rule-Based
_model_cache: dict[str, RetryEngine | None] = {}
_model_warmup_lock = threading.Lock()
_model_warmup_state: dict[str, str] = {
    "status": "idle",
    "message": "models not loaded",
}

_query_cache: dict[tuple[str, str, int], QueryResponse] = {}
_QUERY_CACHE_MAX_ENTRIES = 256

_compare_cache: dict[tuple[str, str, int], CompareQueryResponse] = {}
_COMPARE_CACHE_MAX_ENTRIES = 128


def _is_question_relevant_to_schema(question: str, metadata: DatabaseMetadata) -> bool:
    q = question.lower()

    table_map = metadata.schema.get("tables", {})
    table_names = [name.lower() for name in table_map.keys()]

    column_names: list[str] = []
    for table_name in table_map.keys():
        for col in table_map[table_name].get("columns", []):
            col_name = str(col.get("name", "")).lower()
            if col_name:
                column_names.append(col_name)

    schema_words = set(table_names + column_names)
    if not schema_words:
        return False

    tokens = set(re.findall(r"[a-zA-Z_]+", q))
    if not tokens:
        return False

    overlap = tokens & schema_words
    if overlap:
        return True

    schema_word_list = list(schema_words)
    for token in tokens:
        if get_close_matches(token, schema_word_list, n=1, cutoff=0.8):
            return True

    return False


def _is_complex_question(question: str, metadata: DatabaseMetadata) -> bool:
    """
    Return True if the question is likely complex (requires joins, grouping,
    or explicitly references multiple tables). Simple queries like "list all students"
    or "count all students" return False.
    """
    q = question.lower()

    # Explicit join phrases: "along with", "with their", "and their", etc.
    join_phrases = [
        "along with", "with their", "and their", "associated with", "related to",
        "together with", "including their",
    ]
    if any(phrase in q for phrase in join_phrases):
        return True

    # Grouping phrases: "per department", "each", "by department", etc.
    grouping_phrases = [
        " per ", " each ", "for each", " by ", "in each", "count of",
        "number of students in", "total per", "average per",
    ]
    if any(phrase in q for phrase in grouping_phrases):
        return True

    # Lexical indicators of complex intent (but not "count" or "average" alone)
    complex_markers = [
        " join ", " group by ", " having ", " max(", " min(", " highest ", " lowest ", " top ", " rank", " window", " partition by",
    ]
    if any(marker in q for marker in complex_markers):
        return True

    # Check if question explicitly mentions multiple tables
    try:
        mentioned_tables = set()
        for table in metadata.schema.get("tables", {}).keys():
            table_lower = table.lower()
            singular = table_lower[:-1] if table_lower.endswith("s") else table_lower
            if table_lower in q or singular in q:
                mentioned_tables.add(table)
        # If 2+ tables explicitly mentioned, mark as complex
        if len(mentioned_tables) > 1:
            return True
    except Exception:
        pass

    return False


def _build_model_retry_engine(model_path: Path, model_name: str) -> RetryEngine | None:
    """Build a retry engine for a specific model."""
    setup_adapters(BASE_DIR)
    if LLMSQLGenerator is None:
        logger.warning(f"[MODEL_INIT] LLMSQLGenerator unavailable, cannot load {model_name}")
        return None
    if not model_path.exists():
        logger.warning(f"[MODEL_INIT] Model path does not exist: {model_path}")
        return None
    try:
        logger.info(f"[MODEL_INIT] Loading {model_name} from {model_path}")
        model_generator = LLMSQLGenerator(
            model_path=str(model_path),
            fallback_generator=_rule_gen,
            allow_fallback=False,
        )
        model_generator.warmup()
        logger.info(f"[MODEL_INIT] Successfully loaded {model_name}")
        return RetryEngine(retrieval_service, model_generator, validator, semantic_guard)
    except Exception as exc:
        logger.warning(f"[MODEL_INIT] Failed to load {model_name}: {exc}")
        return None


def _build_rule_retry_engine() -> RetryEngine:
    """Build retry engine for rule-based generator."""
    return RetryEngine(retrieval_service, _rule_gen, validator, semantic_guard)


def _mark_warmup_state(status: str, message: str) -> None:
    _model_warmup_state["status"] = status
    _model_warmup_state["message"] = message
    logger.info(f"[MODEL_STATUS] {status}: {message}")


def _warmup_models() -> None:
    """Eagerly load models at startup."""
    with _model_warmup_lock:
        if _model_warmup_state.get("status") == "ready":
            logger.info("[WARMUP] Models already loaded")
            return
        
        _mark_warmup_state("loading", "initializing models")
        
        try:
            # Load FLAN-T5 first (optimized for basic queries)
            logger.info(f"[WARMUP] Loading FLAN-T5 from {flan_model_dir}")
            _model_cache["flan_t5"] = _build_model_retry_engine(flan_model_dir, "FLAN-T5")
            
            # Then Mistral
            logger.info(f"[WARMUP] Loading Mistral-7B from {mistral_model_dir}")
            _model_cache["mistral_7b"] = _build_model_retry_engine(mistral_model_dir, "Mistral-7B")
            
            flan_loaded = _model_cache.get("flan_t5") is not None
            mistral_loaded = _model_cache.get("mistral_7b") is not None
            logger.info(f"[WARMUP] Results - FLAN-T5: {flan_loaded}, Mistral-7B: {mistral_loaded}")
            
            if flan_loaded or mistral_loaded:
                _mark_warmup_state("ready", "models ready")
            else:
                _mark_warmup_state("partial", "rule-based generator available")
        except Exception as exc:
            logger.error(f"[WARMUP] Error during warmup: {exc}")
            _mark_warmup_state("partial", "rule-based generator available")


def _get_or_init_model_retry_engine(model_name: str) -> RetryEngine | None:
    """Get cached model or initialize it."""
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached
    
    logger.warning(f"[INIT_MODEL] {model_name} not in cache, attempting lazy load")
    if model_name == "flan_t5":
        loaded = _build_model_retry_engine(flan_model_dir, "FLAN-T5")
    elif model_name == "mistral_7b":
        loaded = _build_model_retry_engine(mistral_model_dir, "Mistral-7B")
    else:
        return None
    
    _model_cache[model_name] = loaded
    if loaded is not None:
        _mark_warmup_state("ready", "models ready")
    
    return loaded


def _run_single_model_query(
    *,
    model_name: str,
    model_retry_engine: RetryEngine | None,
    request: QueryRequest,
    metadata: DatabaseMetadata,
    max_attempts: int = 2,
) -> ModelQueryResult:
    """Run a single model's generation pipeline."""
    if model_retry_engine is None:
        return ModelQueryResult(
            model_name=model_name,
            available=False,
            sql="",
            valid=False,
            executed=False,
            attempts=0,
            metrics=QueryMetrics(
                generation_ms=0.0,
                validation_ms=0.0,
                execution_ms=0.0,
                total_ms=0.0,
                attempts=0,
            ),
            error="Model not available",
        )
    
    start_total = time.perf_counter()
    
    try:
        retry_result = model_retry_engine.generate_with_retries(
            question=request.question,
            metadata=metadata,
            few_shot_examples=STORE.list_examples(),
            max_attempts=max_attempts,
        )
        
        sql = retry_result.sql
        valid = len(retry_result.issues) == 0
        
        # Try to execute if valid
        executed = False
        execution_ms = 0.0
        columns: list[str] = []
        rows = []
        row_count = 0
        truncated = False
        execution_error = ""
        
        if valid:
            try:
                start_exec = time.perf_counter()
                result = executor.execute(
                    db_path=str(metadata.file_path) if metadata.file_path is not None else None,
                    sql=sql,
                    max_rows=request.max_rows,
                    db_type=metadata.db_type,
                    connection_url=metadata.connection_url,
                )
                execution_ms = (time.perf_counter() - start_exec) * 1000
                executed = True
                columns = result.columns
                rows = result.rows
                row_count = result.row_count
                truncated = result.truncated
            except Exception as exc:
                logger.warning(f"[EXEC] Execution failed for {model_name}: {exc}")
                execution_error = str(exc)
        
        total_ms = (time.perf_counter() - start_total) * 1000
        
        return ModelQueryResult(
            model_name=model_name,
            available=True,
            sql=sql,
            valid=valid,
            executed=executed,
            attempts=retry_result.attempts,
            metrics=QueryMetrics(
                generation_ms=retry_result.generation_ms,
                validation_ms=retry_result.validation_ms,
                execution_ms=execution_ms,
                total_ms=total_ms,
                attempts=retry_result.attempts,
            ),
            columns=columns,
            rows=rows,
            row_count=row_count,
            truncated=truncated,
            issues=retry_result.issues,
            error=execution_error,
            trace=getattr(retry_result, "trace", None),
        )
    
    except Exception as exc:
        logger.warning(f"[MODEL] {model_name} failed: {exc}")
        return ModelQueryResult(
            model_name=model_name,
            available=True,
            sql="",
            valid=False,
            executed=False,
            attempts=0,
            metrics=QueryMetrics(
                generation_ms=0.0,
                validation_ms=0.0,
                execution_ms=0.0,
                total_ms=(time.perf_counter() - start_total) * 1000,
                attempts=0,
            ),
            error=str(exc),
        )


def _score_result(result: ModelQueryResult) -> tuple:
    """Score a model result for selection."""
    return (
        int(result.executed),  # Executed is most important
        int(result.valid),      # Then valid
        result.attempts,        # Then fewer attempts
        -result.metrics.total_ms,  # Then faster
    )


def _query_response_from_result(
    request: QueryRequest,
    result: ModelQueryResult,
    model_outputs: dict[str, str],
) -> QueryResponse:
    """Build QueryResponse from ModelQueryResult."""
    return QueryResponse(
        db_id=request.db_id,
        question=request.question,
        sql=result.sql,
        valid=result.valid,
        executed=result.executed,
        columns=result.columns or [],
        rows=result.rows or [],
        row_count=result.row_count,
        truncated=result.truncated,
        model_used=result.model_name,
        model_outputs=model_outputs,
        issues=result.issues,
        metrics=result.metrics,
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Load models at startup."""
    setup_adapters(BASE_DIR)
    _warmup_models()


@app.get("/")
def index() -> FileResponse:
    """Serve HTML UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_warmup": _model_warmup_state,
    }


@app.get("/models/status")
def model_status() -> dict:
    """Return current model warmup status for UI polling."""
    return {
        "status": _model_warmup_state.get("status", "idle"),
        "message": _model_warmup_state.get("message", "models not loaded"),
        "models": {
            "flan_t5": _model_cache.get("flan_t5") is not None,
            "mistral_7b": _model_cache.get("mistral_7b") is not None,
        },
    }


@app.post("/models/warmup")
def warmup_models() -> dict:
    """Trigger model warmup (safe to call repeatedly)."""
    _warmup_models()
    return {
        "status": _model_warmup_state.get("status", "idle"),
        "message": _model_warmup_state.get("message", "models not loaded"),
    }


@app.post("/databases/upload", response_model=UploadDatabaseResponse)
async def upload_database(file: UploadFile = File(...)) -> UploadDatabaseResponse:
    """Upload a database file."""
    allowed_extensions = (".db", ".sqlite", ".sqlite3", ".duckdb")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported: .db, .sqlite, .sqlite3, .duckdb",
        )
    
    db_id = str(uuid.uuid4())
    target_path = UPLOAD_DIR / f"{db_id}_{file.filename}"
    
    content = await file.read()
    target_path.write_bytes(content)
    
    db_type = "sqlite"
    connection_url: str | None = None
    
    try:
        if file.filename.lower().endswith(".duckdb"):
            connection_url = f"duckdb:///{target_path.as_posix()}"
            schema, relationships, db_type = schema_service.analyze_database_url(connection_url)
        else:
            schema, relationships = schema_service.analyze_sqlite_database(target_path)
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Database analysis failed: {exc}") from exc
    
    metadata = DatabaseMetadata(
        db_id=db_id,
        name=file.filename,
        source_type="file",
        db_type=db_type,
        file_path=target_path,
        connection_url=connection_url,
        schema=schema,
        relationship_graph=relationships,
        created_at=time.time(),
    )
    STORE.save_database(metadata)
    
    return UploadDatabaseResponse(
        db_id=db_id,
        name=file.filename,
        tables=list(schema["tables"].keys()),
    )


@app.post("/databases/connect", response_model=ConnectDatabaseResponse)
def connect_database(request: ConnectDatabaseRequest) -> ConnectDatabaseResponse:
    """Connect to a remote database."""
    db_id = str(uuid.uuid4())
    try:
        schema, relationships, db_type = schema_service.analyze_database_url(request.db_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {exc}") from exc
    
    metadata = DatabaseMetadata(
        db_id=db_id,
        name=request.name,
        source_type="connection",
        db_type=db_type,
        file_path=None,
        connection_url=request.db_url,
        schema=schema,
        relationship_graph=relationships,
        created_at=time.time(),
    )
    STORE.save_database(metadata)
    
    return ConnectDatabaseResponse(
        db_id=db_id,
        name=request.name,
        tables=list(schema["tables"].keys()),
    )


@app.post("/query", response_model=QueryResponse)
def query_database(request: QueryRequest) -> QueryResponse:
    """Generate and execute SQL for a natural language question.
    
    Pipeline: FLAN-T5 -> Mistral (LLM-only)
    """
    metadata = STORE.get_database(request.db_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Database not found")

    if not _is_question_relevant_to_schema(request.question, metadata):
        logger.warning("[VALIDATION] Question not relevant to database")
        response = QueryResponse(
            db_id=request.db_id,
            question=request.question,
            sql="Question is not relevant to database",
            valid=False,
            executed=False,
            columns=[],
            rows=[],
            row_count=0,
            truncated=False,
            model_used=None,
            issues=["Question is not relevant to database"],
            metrics=QueryMetrics(
                generation_ms=0.0,
                validation_ms=0.0,
                execution_ms=0.0,
                total_ms=0.0,
                attempts=0,
            ),
            model_outputs={},
        )
        return response
    
    # Check cache
    cache_key = (request.db_id, request.question.strip().lower(), request.max_rows)
    cached = _query_cache.get(cache_key)
    if cached is not None:
        return cached
    
    # LLM-only pipeline:
    # - simple query: FLAN-T5 only
    # - complex query: FLAN-T5 -> Mistral
    max_attempts = int(os.getenv("SQL_MAX_ATTEMPTS", "2"))
    # Decide complexity using schema-aware classifier
    is_simple_query = not _is_complex_question(request.question, metadata)
    
    # Initialize FLAN always; initialize Mistral only when needed.
    flan_engine = _get_or_init_model_retry_engine("flan_t5")
    mistral_engine = None if is_simple_query else _get_or_init_model_retry_engine("mistral_7b")
    
    logger.info(f"[PIPELINE] FLAN-T5 engine available: {flan_engine is not None}")
    logger.info(f"[PIPELINE] Mistral-7B engine available: {mistral_engine is not None}")
    logger.info(f"[PIPELINE] Simple query mode: {is_simple_query}")
    
    candidates: list[tuple[str, RetryEngine | None, int]] = [
        ("flan_t5", flan_engine, max_attempts),
    ]
    if not is_simple_query:
        candidates.append(("mistral_7b", mistral_engine, max_attempts))
    
    attempted_results: list[ModelQueryResult] = []
    model_outputs: dict[str, str] = {}
    
    for model_name, model_retry_engine, max_tries in candidates:
        logger.info(f"[PIPELINE] Trying model: {model_name}, engine available: {model_retry_engine is not None}")
        
        result = _run_single_model_query(
            model_name=model_name,
            model_retry_engine=model_retry_engine,
            request=request,
            metadata=metadata,
            max_attempts=max_tries,
        )
        attempted_results.append(result)
        # Include raw LLM output and postprocessed/final SQL for debugging
        trace = result.trace or {}
        raw = trace.get("decoded", "") if isinstance(trace, dict) else ""
        extracted = trace.get("extracted", "") if isinstance(trace, dict) else ""
        postprocessed = trace.get("postprocessed", "") if isinstance(trace, dict) else ""
        model_outputs[model_name] = (
            f"RAW:\n{raw}\n\nEXTRACTED:\n{extracted}\n\nPOSTPROCESSED/FINAL:\n{postprocessed or result.sql}"
        )
        
        logger.info(f"[PIPELINE] {model_name} result: executed={result.executed}, valid={result.valid}, sql_len={len(result.sql)}")
        logger.info(f"[PIPELINE] {model_name} final SQL: {result.sql}")
    
    # Try all models, then pick the best result instead of early exit
    # This ensures Mistral gets tried even if FLAN's structural repair passes validation
    
    # If no model succeeded, pick the best attempt
    executed_results = [r for r in attempted_results if r.executed]
    if executed_results:
        best = max(executed_results, key=_score_result)
    else:
        best = max(attempted_results, key=_score_result)
    
    response = _query_response_from_result(
        request=request,
        result=best,
        model_outputs=model_outputs,
    )
    
    if len(_query_cache) >= _QUERY_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(_query_cache))
        _query_cache.pop(oldest_key, None)
    _query_cache[cache_key] = response
    
    return response


@app.post("/query/compare", response_model=CompareQueryResponse)
def compare_query_models(request: QueryRequest) -> CompareQueryResponse:
    """Compare FLAN-T5 and Mistral on a query."""
    metadata = STORE.get_database(request.db_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Database not found")

    if not _is_question_relevant_to_schema(request.question, metadata):
        logger.warning("[VALIDATION] Question not relevant to database")
        empty_metrics = QueryMetrics(
            generation_ms=0.0,
            validation_ms=0.0,
            execution_ms=0.0,
            total_ms=0.0,
            attempts=0,
        )
        invalid_result = ModelQueryResult(
            model_name="none",
            available=True,
            sql="Question is not relevant to database",
            valid=False,
            executed=False,
            attempts=0,
            metrics=empty_metrics,
            issues=["Question is not relevant to database"],
        )
        return CompareQueryResponse(
            db_id=request.db_id,
            question=request.question,
            winner_model=None,
            winner_reason="Question is not relevant to database",
            models={
                "flan_t5": invalid_result,
                "mistral_7b": invalid_result,
            },
        )
    
    cache_key = (request.db_id, request.question.strip().lower(), request.max_rows)
    cached = _compare_cache.get(cache_key)
    if cached is not None:
        return cached
    
    compare_attempts = int(os.getenv("SQL_COMPARE_MAX_ATTEMPTS", "2"))
    
    flan_engine = _get_or_init_model_retry_engine("flan_t5")
    mistral_engine = _get_or_init_model_retry_engine("mistral_7b")
    
    flan_result = _run_single_model_query(
        model_name="flan_t5",
        model_retry_engine=flan_engine,
        request=request,
        metadata=metadata,
        max_attempts=compare_attempts,
    )
    
    mistral_result = _run_single_model_query(
        model_name="mistral_7b",
        model_retry_engine=mistral_engine,
        request=request,
        metadata=metadata,
        max_attempts=compare_attempts,
    )
    
    # Pick winner based on execution and validity
    if flan_result.executed and not mistral_result.executed:
        winner = "flan_t5"
        reason = "FLAN-T5 executed successfully"
    elif mistral_result.executed and not flan_result.executed:
        winner = "mistral_7b"
        reason = "Mistral executed successfully"
    elif flan_result.executed and mistral_result.executed:
        if _score_result(flan_result) > _score_result(mistral_result):
            winner = "flan_t5"
            reason = "FLAN-T5 had better metrics"
        else:
            winner = "mistral_7b"
            reason = "Mistral had better metrics"
    else:
        if flan_result.valid and not mistral_result.valid:
            winner = "flan_t5"
            reason = "FLAN-T5 SQL was valid"
        elif mistral_result.valid and not flan_result.valid:
            winner = "mistral_7b"
            reason = "Mistral SQL was valid"
        else:
            winner = "flan_t5" if _score_result(flan_result) > _score_result(mistral_result) else "mistral_7b"
            reason = "Better overall score"
    
    response = CompareQueryResponse(
        db_id=request.db_id,
        question=request.question,
        winner_model=winner,
        winner_reason=reason,
        models={
            "flan_t5": flan_result,
            "mistral_7b": mistral_result,
        },
    )
    
    if len(_compare_cache) >= _COMPARE_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(_compare_cache))
        _compare_cache.pop(oldest_key, None)
    _compare_cache[cache_key] = response
    
    return response


@app.get("/databases/{db_id}/schema")
def get_schema(db_id: str) -> dict:
    """Get database schema."""
    metadata = STORE.get_database(db_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Database not found")
    
    return {
        "db_id": metadata.db_id,
        "name": metadata.name,
        "schema": metadata.schema,
        "relationships": metadata.relationship_graph,
    }