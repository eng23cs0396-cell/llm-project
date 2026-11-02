from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal

from app.services.prompt_engine import PromptEngine

logger = logging.getLogger(__name__)


def extract_sql_from_text(text: str) -> str:
    """Extract clean SQL query from LLM output, handling markdown blocks and explanations."""
    if not text:
        return ""

    text = text.strip()

    # 1. Match ```sql ... ``` block
    sql_block = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if sql_block:
        candidate = sql_block.group(1).strip()
        # Drop leading non-sql words if any
        candidate = re.sub(r"^(?:sql:?|query:?)\s*", "", candidate, flags=re.IGNORECASE)
        return candidate.rstrip(";")

    # 2. If starts with SELECT or WITH
    sql_match = re.search(r"(?i)\b(SELECT|WITH)\b[\s\S]+", text)
    if sql_match:
        sql = sql_match.group(0).strip()
        # Cut off any trailing explanation after semicolon
        if ";" in sql:
            sql = sql.split(";")[0]
        return sql.strip()

    return text.strip().rstrip(";")


class NIMSQLService:
    """NVIDIA NIM Text-to-SQL Service powered by LangChain."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        primary_model: str = "meta/llama-3.3-70b-instruct",
        secondary_model: str = "qwen/qwen2.5-coder-32b-instruct",
        local_model_url: str = "http://localhost:11434/v1",
        local_model_name: str = "llama3:latest",
        prompt_engine: PromptEngine | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "").strip()
        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.primary_model = primary_model or os.getenv("NVIDIA_PRIMARY_MODEL", "meta/llama-3.3-70b-instruct")
        self.secondary_model = secondary_model or os.getenv("NVIDIA_SECONDARY_MODEL", "qwen/qwen2.5-coder-32b-instruct")
        self.local_model_url = local_model_url or os.getenv("LOCAL_MODEL_URL", "http://localhost:11434/v1")
        self.local_model_name = local_model_name or os.getenv("LOCAL_MODEL_NAME", "llama3:latest")
        self.prompt_engine = prompt_engine or PromptEngine()

        self._llm_instances: dict[str, Any] = {}

    def is_configured(self) -> bool:
        """Returns True if an NVIDIA API key is available."""
        return bool(self.api_key and self.api_key.startswith("nvapi-"))

    def is_local_available(self) -> bool:
        """Returns True if local Ollama/OpenAI-compatible server is reachable."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.local_model_url}/models", headers={"User-Agent": "NL2SQL"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_model_name(self, model_choice: str | None = None) -> str:
        """Resolve model name from user choice or defaults."""
        choice = (model_choice or os.getenv("DEFAULT_MODEL_CHOICE", "auto")).lower().strip()
        if choice in ("local", "ollama", "local_ollama", "local_model"):
            return self.local_model_name
        if choice in ("secondary", "coder", "qwen", "qwen2.5-coder-32b-instruct"):
            return self.secondary_model
        if choice in ("primary", "llama", "llama-3.3-70b-instruct", "auto"):
            return self.primary_model
        return model_choice if model_choice else self.primary_model

    def _get_langchain_client(self, model_name: str):
        """Lazy-initialize LangChain ChatOpenAI client pointing to NVIDIA NIM or Local Ollama."""
        is_local = (model_name == self.local_model_name)
        target_base_url = self.local_model_url if is_local else self.base_url
        target_api_key = "ollama" if is_local else self.api_key

        if not is_local and not self.is_configured():
            raise ValueError(
                "NVIDIA API Key is missing. Set NVIDIA_API_KEY in your .env file with a key starting with 'nvapi-'."
            )

        client_key = f"{target_base_url}:{model_name}"
        if client_key not in self._llm_instances:
            try:
                from langchain_openai import ChatOpenAI
                self._llm_instances[client_key] = ChatOpenAI(
                    model=model_name,
                    openai_api_key=target_api_key,
                    openai_api_base=target_base_url,
                    temperature=0.0,
                    max_tokens=2048,
                    timeout=120.0,
                    max_retries=2,
                )
            except ImportError:
                from openai import OpenAI
                client = OpenAI(base_url=target_base_url, api_key=target_api_key, timeout=120.0)
                self._llm_instances[client_key] = client

        return self._llm_instances[client_key]

    def generate_sql(
        self,
        question: str,
        schema: dict[str, Any],
        dialect: str = "sqlite",
        model_choice: str | None = None,
        few_shot_k: int = 3,
    ) -> dict[str, Any]:
        """Generate SQL for a question using dynamic few-shot prompting and NVIDIA NIM."""
        resolved_model = self.get_model_name(model_choice)
        logger.info("[NIM_SERVICE] Generating SQL with model %s for question: %s", resolved_model, question)

        schema_text = self.prompt_engine.format_schema_context(schema, dialect)
        few_shots = self.prompt_engine.select_few_shot_examples(question, k=few_shot_k)

        system_msg = self.prompt_engine.build_system_prompt(dialect)
        user_msg = self.prompt_engine.build_user_prompt(question, schema_text, few_shots)

        llm = self._get_langchain_client(resolved_model)

        # Execute call via LangChain if available, else OpenAI fallback
        if hasattr(llm, "invoke"):
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ]
            response = llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
        else:
            # Fallback to standard OpenAI client
            response = llm.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content

        clean_sql = extract_sql_from_text(raw_text)

        return {
            "sql": clean_sql,
            "raw_output": raw_text,
            "model_used": resolved_model,
            "few_shot_count": len(few_shots),
            "few_shot_categories": [ex.category for ex in few_shots],
        }

    def repair_sql(
        self,
        question: str,
        failed_sql: str,
        error_message: str,
        schema: dict[str, Any],
        dialect: str = "sqlite",
        model_choice: str | None = None,
    ) -> dict[str, Any]:
        """Self-healing reflection: Feed database execution error back to the model to correct the SQL."""
        resolved_model = self.get_model_name(model_choice)
        logger.info("[NIM_SERVICE] Repairing failed SQL with model %s. Error: %s", resolved_model, error_message)

        schema_text = self.prompt_engine.format_schema_context(schema, dialect)
        system_msg = self.prompt_engine.build_system_prompt(dialect)
        user_msg = self.prompt_engine.build_repair_prompt(
            question=question,
            failed_sql=failed_sql,
            error_message=error_message,
            schema_text=schema_text,
            dialect=dialect,
        )

        llm = self._get_langchain_client(resolved_model)

        if hasattr(llm, "invoke"):
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ]
            response = llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
        else:
            response = llm.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content

        repaired_sql = extract_sql_from_text(raw_text)

        return {
            "sql": repaired_sql,
            "raw_output": raw_text,
            "model_used": resolved_model,
            "was_repaired": True,
        }
