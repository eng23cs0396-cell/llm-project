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
from app.services.prompt_engine import PromptEngine
from app.services.nim_service import NIMSQLService

_rule_gen = SQLGenerator()
prompt_engine = PromptEngine()
nim_service = NIMSQLService(prompt_engine=prompt_engine)

app = FastAPI(title="NL-to-SQL Service", version="0.2.0")

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
        
        if nim_service.is_configured():
            logger.info(f"[WARMUP] NVIDIA NIM is configured ({nim_service.primary_model})")
            _mark_warmup_state("ready", f"NVIDIA NIM Ready ({nim_service.primary_model})")
            return

        try:
            # Load FLAN-T5 first (optimized for basic queries)
            if flan_model_dir.exists():
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


def _run_nim_model_query(
    *,
    model_choice: str,
    request: QueryRequest,
    metadata: DatabaseMetadata,
    few_shot_k: int = 3,
) -> ModelQueryResult:
    """Run generation pipeline using NVIDIA NIM API with dynamic few-shot prompting and self-healing reflection."""
    start_total = time.perf_counter()
    resolved_model_name = nim_service.get_model_name(model_choice)

    is_local = model_choice.lower() in ("local", "ollama", "local_ollama", "local_model")
    if is_local:
        if not nim_service.is_local_available():
            return ModelQueryResult(
                model_name=resolved_model_name,
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
                error=f"Local model endpoint not reachable at {nim_service.local_model_url}",
            )
    elif not nim_service.is_configured():
        return ModelQueryResult(
            model_name=resolved_model_name,
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
            error="NVIDIA API Key not configured in .env",
        )

    try:
        start_gen = time.perf_counter()
        gen_res = nim_service.generate_sql(
            question=request.question,
            schema=metadata.schema,
            dialect=metadata.db_type,
            model_choice=model_choice,
            few_shot_k=few_shot_k,
        )
        generation_ms = (time.perf_counter() - start_gen) * 1000
        sql = gen_res.get("sql", "").strip()

        start_val = time.perf_counter()
        val_issues = validator.validate(sql, metadata.schema, dialect=metadata.db_type)
        validation_ms = (time.perf_counter() - start_val) * 1000
        issue_messages = [i.message for i in val_issues if i.level == "error"]

        executed = False
        execution_ms = 0.0
        columns = []
        rows = []
        row_count = 0
        truncated = False
        execution_error = ""
        attempts = 1

        if sql and not issue_messages:
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
                execution_error = str(exc)
                logger.info(f"[NIM] Execution failed ({exc}), attempting self-healing reflection...")

                # Self-healing reflection loop
                try:
                    attempts = 2
                    repaired = nim_service.repair_sql(
                        question=request.question,
                        failed_sql=sql,
                        error_message=execution_error,
                        schema=metadata.schema,
                        dialect=metadata.db_type,
                        model_choice=model_choice,
                    )
                    sql = repaired.get("sql", sql).strip()
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
                    execution_error = ""
                    logger.info("[NIM] Self-healing reflection succeeded! Corrected SQL executed successfully.")
                except Exception as repair_exc:
                    execution_error = str(repair_exc)
                    logger.warning(f"[NIM] Self-healing reflection failed: {repair_exc}")

        total_ms = (time.perf_counter() - start_total) * 1000

        return ModelQueryResult(
            model_name=resolved_model_name,
            available=True,
            sql=sql,
            valid=executed or (len(issue_messages) == 0 and bool(sql)),
            executed=executed,
            attempts=attempts,
            metrics=QueryMetrics(
                generation_ms=generation_ms,
                validation_ms=validation_ms,
                execution_ms=execution_ms,
                total_ms=total_ms,
                attempts=attempts,
            ),
            columns=columns,
            rows=rows,
            row_count=row_count,
            truncated=truncated,
            issues=issue_messages,
            error=execution_error or None,
            trace={
                "raw_output": gen_res.get("raw_output", ""),
                "few_shot_categories": gen_res.get("few_shot_categories", []),
            },
        )
    except Exception as exc:
        logger.error(f"[NIM] Query generation exception: {exc}")
        return ModelQueryResult(
            model_name=resolved_model_name,
            available=False,
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
        "nvidia_configured": nim_service.is_configured(),
        "models": {
            "nvidia_primary": nim_service.primary_model,
            "nvidia_secondary": nim_service.secondary_model,
            "flan_t5": _model_cache.get("flan_t5") is not None,
            "rule_based": True,
        },
        "default_model": os.getenv("DEFAULT_MODEL_CHOICE", "auto"),
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
    few_shot_k = int(os.getenv("FEW_SHOT_MAX_EXAMPLES", "3"))
    chosen_model = (request.model or os.getenv("DEFAULT_MODEL_CHOICE", "auto")).lower().strip()

    attempted_results: list[ModelQueryResult] = []
    model_outputs: dict[str, str] = {}

    # 1. Try Cloud / Local LLM (NVIDIA NIM or Local Ollama)
    use_nim = False
    nim_choice = "primary"
    if chosen_model in ("local", "ollama", "local_ollama", "local_model"):
        use_nim = True
        nim_choice = "local"
    elif chosen_model in ("primary", "nvidia_primary", "llama", "llama-3.3-70b", "meta/llama-3.3-70b-instruct"):
        use_nim = True
        nim_choice = "primary"
    elif chosen_model in ("secondary", "nvidia_secondary", "qwen", "coder", "qwen2.5-coder-32b-instruct"):
        use_nim = True
        nim_choice = "secondary"
    elif chosen_model == "auto":
        if nim_service.is_configured():
            use_nim = True
            nim_choice = "primary"
        elif nim_service.is_local_available():
            use_nim = True
            nim_choice = "local"

    if use_nim and (nim_service.is_configured() or nim_choice == "local"):
        logger.info(f"[PIPELINE] Attempting LLM service with choice: {nim_choice}")
        nim_res = _run_nim_model_query(
            model_choice=nim_choice,
            request=request,
            metadata=metadata,
            few_shot_k=few_shot_k,
        )
        attempted_results.append(nim_res)
        model_outputs[nim_res.model_name] = (
            f"RAW:\n{(nim_res.trace or {}).get('raw_output', '')}\n\nFINAL SQL:\n{nim_res.sql}"
        )

        # If primary succeeded, return directly
        if nim_res.executed:
            response = _query_response_from_result(
                request=request,
                result=nim_res,
                model_outputs=model_outputs,
            )
            _query_cache[cache_key] = response
            return response

        # If primary failed and auto mode is active, try secondary NIM model
        if chosen_model == "auto" and not nim_res.executed:
            logger.info("[PIPELINE] Primary NIM failed, trying secondary NVIDIA NIM model...")
            nim_sec_res = _run_nim_model_query(
                model_choice="secondary",
                request=request,
                metadata=metadata,
                few_shot_k=few_shot_k,
            )
            attempted_results.append(nim_sec_res)
            model_outputs[nim_sec_res.model_name] = (
                f"RAW:\n{(nim_sec_res.trace or {}).get('raw_output', '')}\n\nFINAL SQL:\n{nim_sec_res.sql}"
            )
            if nim_sec_res.executed:
                response = _query_response_from_result(
                    request=request,
                    result=nim_sec_res,
                    model_outputs=model_outputs,
                )
                _query_cache[cache_key] = response
                return response

    # 2. Local Model Fallback (FLAN-T5)
    flan_engine = _get_or_init_model_retry_engine("flan_t5")
    if flan_engine is not None:
        logger.info("[PIPELINE] Running Local FLAN-T5 engine...")
        flan_res = _run_single_model_query(
            model_name="flan_t5",
            model_retry_engine=flan_engine,
            request=request,
            metadata=metadata,
            max_attempts=max_attempts,
        )
        attempted_results.append(flan_res)
        trace = flan_res.trace or {}
        model_outputs["flan_t5"] = f"EXTRACTED:\n{trace.get('extracted', '')}\n\nFINAL SQL:\n{flan_res.sql}"

    # 3. Rule-based generator fallback if nothing executed yet
    if not any(r.executed for r in attempted_results):
        logger.info("[PIPELINE] Running rule-based fallback generator...")
        rule_engine = _build_rule_retry_engine()
        rule_res = _run_single_model_query(
            model_name="rule_based",
            model_retry_engine=rule_engine,
            request=request,
            metadata=metadata,
            max_attempts=1,
        )
        attempted_results.append(rule_res)
        model_outputs["rule_based"] = f"FINAL SQL:\n{rule_res.sql}"

    # Pick the best result
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
    """Compare multiple models on a query (NVIDIA NIM Primary, Secondary, and Local FLAN-T5)."""
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
                "nvidia_primary": invalid_result,
                "nvidia_secondary": invalid_result,
            },
        )

    cache_key = (request.db_id, request.question.strip().lower(), request.max_rows)
    cached = _compare_cache.get(cache_key)
    if cached is not None:
        return cached

    few_shot_k = int(os.getenv("FEW_SHOT_MAX_EXAMPLES", "3"))
    compared_models: dict[str, ModelQueryResult] = {}

    if nim_service.is_configured():
        # Compare NVIDIA Primary vs NVIDIA Secondary
        nim_p = _run_nim_model_query(
            model_choice="primary",
            request=request,
            metadata=metadata,
            few_shot_k=few_shot_k,
        )
        compared_models["nvidia_primary"] = nim_p

        nim_s = _run_nim_model_query(
            model_choice="secondary",
            request=request,
            metadata=metadata,
            few_shot_k=few_shot_k,
        )
        compared_models["nvidia_secondary"] = nim_s
    else:
        # Fallback to local models when NIM is not configured
        flan_engine = _get_or_init_model_retry_engine("flan_t5")
        flan_result = _run_single_model_query(
            model_name="flan_t5",
            model_retry_engine=flan_engine,
            request=request,
            metadata=metadata,
            max_attempts=2,
        )
        compared_models["flan_t5"] = flan_result

        rule_engine = _build_rule_retry_engine()
        rule_result = _run_single_model_query(
            model_name="rule_based",
            model_retry_engine=rule_engine,
            request=request,
            metadata=metadata,
            max_attempts=1,
        )
        compared_models["rule_based"] = rule_result

    # Pick winner
    valid_candidates = [m for m in compared_models.values() if m.available]
    if not valid_candidates:
        winner = None
        reason = "No models available"
    else:
        best_candidate = max(valid_candidates, key=_score_result)
        winner = best_candidate.model_name
        if best_candidate.executed:
            reason = f"{winner} executed successfully with lowest latency / attempts"
        elif best_candidate.valid:
            reason = f"{winner} generated valid SQL syntax"
        else:
            reason = f"{winner} had the highest validation score"

    response = CompareQueryResponse(
        db_id=request.db_id,
        question=request.question,
        winner_model=winner,
        winner_reason=reason,
        models=compared_models,
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