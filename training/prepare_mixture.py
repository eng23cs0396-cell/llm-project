from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


def try_load_spider(sample_count: int | None = None) -> Dataset:
    candidates = [
        ("spider", None),
        ("xlangai/spider", None),
        ("Salesforce/spider", None),
    ]
    split = "train"
    if sample_count and sample_count > 0:
        split = f"train[:{sample_count}]"
    for name, subset in candidates:
        try:
            if subset:
                ds = load_dataset(name, subset, split=split)
            else:
                ds = load_dataset(name, split=split)
            return ds
        except Exception:
            continue
    raise RuntimeError(
        "Unable to load Spider from Hugging Face. Provide local Spider JSONL via --spider-local"
    )


def try_load_wikisql(sample_count: int | None = None) -> Dataset:
    split = "train"
    if sample_count and sample_count > 0:
        split = f"train[:{sample_count}]"
    return load_dataset("wikisql", split=split, trust_remote_code=True)


def try_load_bird_hf(dataset_name: str, sample_count: int | None = None) -> Dataset:
    split = "train"
    if sample_count and sample_count > 0:
        split = f"train[:{sample_count}]"
    return load_dataset(dataset_name, split=split)


def load_bird_local(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"BIRD file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_spider_row(row: dict[str, Any]) -> dict[str, str]:
    question = row.get("question") or row.get("utterance") or ""
    sql = row.get("query") or row.get("sql") or ""
    schema_hint = ""

    if "db_id" in row:
        schema_hint = f"Database ID: {row['db_id']}"

    return {
        "instruction": "Generate a valid SQL query for the question.",
        "input": f"Question: {question}\n{schema_hint}".strip(),
        "output": sql,
        "source": "spider",
    }


def normalize_wikisql_row(row: dict[str, Any]) -> dict[str, str]:
    question = row.get("question", "")
    sql = row.get("sql", "")
    sql_text = sql if isinstance(sql, str) else str(sql)

    table = row.get("table", {})
    headers = table.get("header", []) if isinstance(table, dict) else []
    schema_hint = f"Columns: {', '.join(headers)}" if headers else ""

    return {
        "instruction": "Generate a valid SQL query for the question.",
        "input": f"Question: {question}\n{schema_hint}".strip(),
        "output": sql_text,
        "source": "wikisql",
    }


def normalize_bird_row(row: dict[str, Any]) -> dict[str, str]:
    question = row.get("question") or row.get("instruction") or ""
    sql = row.get("SQL") or row.get("sql") or row.get("query") or row.get("output") or ""
    db_id = row.get("db_id") or row.get("db") or ""
    schema_hint = f"Database ID: {db_id}" if db_id else ""

    return {
        "instruction": "Generate a valid SQL query for the question.",
        "input": f"Question: {question}\n{schema_hint}".strip(),
        "output": sql,
        "source": "bird",
    }


def sample_dataset(ds: list[dict[str, str]], n: int | None, seed: int) -> list[dict[str, str]]:
    if n is None or n <= 0 or n >= len(ds):
        return ds
    rng = random.Random(seed)
    return rng.sample(ds, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Spider + WikiSQL + BIRD mixture")
    parser.add_argument("--out", type=Path, default=Path("training/data/mixture_train.jsonl"))
    parser.add_argument("--spider-local", type=Path, default=None)
    parser.add_argument("--bird-hf", type=str, default="birdsql/bird23-train-filtered")
    parser.add_argument("--bird-local", type=Path, default=None)
    parser.add_argument("--spider-samples", type=int, default=12000)
    parser.add_argument("--wikisql-samples", type=int, default=18000)
    parser.add_argument("--bird-samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.spider_local:
        spider_rows = load_bird_local(args.spider_local)
        spider_norm = [normalize_spider_row(row) for row in spider_rows]
    else:
        spider_ds = try_load_spider(args.spider_samples)
        spider_norm = [normalize_spider_row(row) for row in spider_ds]

    wikisql_ds = try_load_wikisql(args.wikisql_samples)
    wikisql_norm = [normalize_wikisql_row(row) for row in wikisql_ds]

    bird_norm: list[dict[str, str]] = []
    if args.bird_local:
        bird_rows = load_bird_local(args.bird_local)
        bird_norm = [normalize_bird_row(row) for row in bird_rows]
    else:
        try:
            bird_ds = try_load_bird_hf(args.bird_hf, args.bird_samples)
            bird_norm = [normalize_bird_row(row) for row in bird_ds]
        except Exception as exc:
            print(f"[WARN] Could not load BIRD from Hugging Face ({args.bird_hf}): {exc}")
            print("[WARN] Continuing without BIRD. Pass --bird-local or a working --bird-hf to include it.")

    spider_norm = sample_dataset(spider_norm, args.spider_samples, args.seed)
    wikisql_norm = sample_dataset(wikisql_norm, args.wikisql_samples, args.seed)
    bird_norm = sample_dataset(bird_norm, args.bird_samples, args.seed)

    all_rows = spider_norm + wikisql_norm + bird_norm
    random.Random(args.seed).shuffle(all_rows)

    with args.out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(all_rows)} rows to {args.out}")
    print(
        "Breakdown:",
        {
            "spider": len(spider_norm),
            "wikisql": len(wikisql_norm),
            "bird": len(bird_norm),
        },
    )


if __name__ == "__main__":
    main()
