"""Ablation retrieval/chunking chỉ chạy trên tập dev."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IndexConfig
from src.evaluate import calculate_retrieval_metrics, load_benchmark
from src.index import build_index
from src.retrieval import Retriever
from src.security import AccessContext
from src.utils import save_json, setup_logging

LOGGER = logging.getLogger("rag_knowledge_assistant.ablation")
DEV_ACCESS = AccessContext(
    user_id="benchmark-dev",
    groups=("employee", "hr", "finance", "security"),
    auth_method="offline-benchmark",
)


def _retrieval_rows(
    retriever: Retriever,
    cases: list[Any],
    use_dense: bool,
    use_bm25: bool,
    use_reranker: bool,
) -> dict[str, Any]:
    answerable = [case for case in cases if case.is_answerable]
    ranked: list[list[str]] = []
    expected: list[set[str]] = []
    latencies: list[float] = []
    for case in answerable:
        started = time.perf_counter()
        hits = retriever.search(
            case.question,
            k=4,
            use_dense=use_dense,
            use_bm25=use_bm25,
            use_reranker=use_reranker,
            access_context=DEV_ACCESS,
            # Giữ cả logit thô âm để không làm sai thứ hạng baseline.
            min_score=float("-inf"),
        )
        latencies.append(time.perf_counter() - started)
        ranked.append([str(hit.get("source", "")) for hit in hits])
        expected.append(set(case.expected_documents or case.expected_sources))
    metrics = calculate_retrieval_metrics(ranked, expected, latencies, k=4)
    return {
        "recall_at_k": metrics["recall_at_k"],
        "hit_rate_at_1": metrics["hit_rate_at_1"],
        "mrr": metrics["mrr"],
        "ndcg_at_k": metrics["ndcg_at_k"],
        "avg_latency_ms": metrics["avg_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],
    }


def run_rag_ablation(
    retriever: Retriever, dev_cases: list[Any], output_path: Path
) -> dict[str, Any]:
    """So sánh các pipeline đã định nghĩa trước trên DEV, không sweep Test."""
    configurations = [
        {"name": "BM25 Only", "use_dense": False, "use_bm25": True, "use_reranker": False},
        {"name": "Dense Only", "use_dense": True, "use_bm25": False, "use_reranker": False},
        {
            "name": "Dense + BM25 + RRF",
            "use_dense": True,
            "use_bm25": True,
            "use_reranker": False,
        },
        {
            "name": "RRF + Multilingual Reranker",
            "use_dense": True,
            "use_bm25": True,
            "use_reranker": True,
        },
    ]
    results = []
    for configuration in configurations:
        row = _retrieval_rows(
            retriever,
            dev_cases,
            configuration["use_dense"],
            configuration["use_bm25"],
            configuration["use_reranker"],
        )
        results.append({"pipeline": configuration["name"], **row})
    report = {
        "selection_split": "dev",
        "locked_test_accessed": False,
        "results": results,
    }
    save_json(output_path, report)
    return report


def run_chunk_ablation(
    data_dir: str,
    temp_dir: Path,
    dev_cases: list[Any],
    output_path: Path,
) -> dict[str, Any]:
    """Chọn chunk config trên DEV; không dùng locked test để chọn."""
    chunk_configs = [
        ("Sliding Window 120/20", "sliding_window", 120, 20),
        ("Sliding Window 220/30", "sliding_window", 220, 30),
        ("Sliding Window 350/50", "sliding_window", 350, 50),
        ("Structure-Aware 250/40", "structure_aware", 250, 40),
    ]
    results = []
    for name, strategy, chunk_words, overlap_words in chunk_configs:
        model_dir = temp_dir / f"{strategy}_{chunk_words}"
        config = IndexConfig(
            data_dir=data_dir,
            model_dir=str(model_dir),
            catalog_path="configs/knowledge_catalog.yaml",
            strategy=strategy,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )
        build_index(config)
        retriever = Retriever(model_dir=model_dir, use_reranker=False)
        row = _retrieval_rows(retriever, dev_cases, True, True, False)
        results.append({"strategy": name, "total_chunks": len(retriever.chunks), **row})
    report = {
        "selection_split": "dev",
        "locked_test_accessed": False,
        "results": results,
    }
    save_json(output_path, report)
    return report


def main() -> None:
    setup_logging()
    benchmark_path = PROJECT_ROOT / "data/evaluation/synthetic_regression.json"
    if not benchmark_path.exists():
        benchmark_path = PROJECT_ROOT / "data/evaluation/questions.json"
    cases = load_benchmark(benchmark_path)
    dev_cases = [case for case in cases if case.split == "dev"]
    retriever = Retriever(model_dir=PROJECT_ROOT / "artifacts", use_reranker=True)
    run_rag_ablation(retriever, dev_cases, PROJECT_ROOT / "reports/rag_ablation.json")
    run_chunk_ablation(
        "data/raw",
        PROJECT_ROOT / "reports/ablation_scratch",
        dev_cases,
        PROJECT_ROOT / "reports/chunk_ablation.json",
    )


if __name__ == "__main__":
    main()
