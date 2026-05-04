from __future__ import annotations

import re
from collections import Counter


class EmbeddingService:
    """
    Lightweight lexical-semantic scorer.

    This keeps the architecture plug-compatible with vector embeddings while
    remaining dependency-light for the initial phase.
    """

    token_pattern = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

    def score_similarity(self, query: str, candidates: list[str]) -> list[tuple[str, float]]:
        query_vec = self._vectorize(query)
        results: list[tuple[str, float]] = []
        for candidate in candidates:
            score = self._cosine(query_vec, self._vectorize(candidate))
            results.append((candidate, score))
        return sorted(results, key=lambda item: item[1], reverse=True)

    def _vectorize(self, text: str) -> Counter[str]:
        tokens = [token.lower() for token in self.token_pattern.findall(text)]
        return Counter(tokens)

    def _cosine(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0

        dot = sum(left[token] * right[token] for token in set(left) & set(right))
        left_norm = sum(value * value for value in left.values()) ** 0.5
        right_norm = sum(value * value for value in right.values()) ** 0.5

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
