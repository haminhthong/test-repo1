"""Script tự động hóa các nghiên cứu thực nghiệm bóc tách (Ablation Experiments).

Thực hiện 2 nghiên cứu bóc tách chuẩn mực AI Engineering:
1. RAG Pipeline Ablation:
   - Dense Only
   - BM25 Only
   - Dense + BM25 (Linear Fusion)
   - Dense + BM25 (RRF)
   - Dense + BM25 + RRF + Cross-Encoder Reranker
   - Canonical RAG (+ Evidence Gate)
2. Chunking Strategy Ablation:
   - Sliding Window: 120 words / 20 overlap
   - Sliding Window: 220 words / 30 overlap
   - Sliding Window: 350 words / 50 overlap
   - Structure-Aware Chunking (Sentence Boundaries & Heading Packing)
"""

from __future__ import annotations

import logging
import time
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IndexConfig
from src.evaluate import calculate_retrieval_metrics, load_benchmark
from src.index import build_index
from src.retrieval import Retriever
from src.utils import save_json, setup_logging

LOGGER = logging.getLogger("rag_knowledge_assistant.ablation")


def run_rag_ablation(
    retriever: Retriever,
    test_cases: list[Any],
    output_path: Path,
) -> dict[str, Any]:
    """Chạy ablation các thành phần trong RAG pipeline."""
    LOGGER.info("--- Bắt đầu RAG Component Ablation ---")

    ans_cases = [c for c in test_cases if c.is_answerable]
    expected_sources = [set(c.expected_documents or c.expected_sources) for c in ans_cases]

    configurations = [
        {"name": "1. Dense Only (FAISS)", "dense_weight": 1.0, "use_reranker": False, "use_gate": False},
        {"name": "2. BM25 Only (BM25Okapi)", "dense_weight": 0.0, "use_reranker": False, "use_gate": False},
        {"name": "3. Dense + BM25 (Linear Fusion)", "dense_weight": 0.85, "use_reranker": False, "use_gate": False},
        {"name": "4. Dense + BM25 (RRF Union)", "dense_weight": None, "use_reranker": False, "use_gate": False},
        {"name": "5. RRF + Cross-Encoder Reranker", "dense_weight": None, "use_reranker": True, "use_gate": False},
        {"name": "6. Canonical RAG (+ Evidence Gate)", "dense_weight": None, "use_reranker": True, "use_gate": True},
    ]

    ablation_results: list[dict[str, Any]] = []

    for cfg in configurations:
        name = cfg["name"]
        dw = cfg["dense_weight"]
        rerank = cfg["use_reranker"]
        gate = cfg["use_gate"]

        ranked_docs: list[list[str]] = []
        latencies: list[float] = []

        for case in ans_cases:
            t0 = time.perf_counter()
            hits = retriever.search(
                case.question,
                k=4,
                dense_weight=dw,
                use_reranker=rerank,
            )
            latencies.append(time.perf_counter() - t0)

            if gate:
                # Lọc các kết quả không vượt qua Evidence Gate
                valid_hits = [h for h in hits if h.get("gate_passed", True)]
            else:
                valid_hits = hits

            ranked_docs.append([str(h.get("source", "")) for h in valid_hits])

        metrics = calculate_retrieval_metrics(ranked_docs, expected_sources, latencies)
        result_item = {
            "component": name,
            "recall_at_k": metrics["recall_at_k"],
            "hit_rate_at_1": metrics["hit_rate_at_1"],
            "mrr": metrics["mrr"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "avg_latency_ms": metrics["avg_latency_ms"],
            "p95_latency_ms": metrics["p95_latency_ms"],
        }
        ablation_results.append(result_item)

    report = {
        "title": "RAG Component Ablation Study",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": len(ans_cases),
        "results": ablation_results,
    }
    save_json(output_path, report)
    LOGGER.info("Đã lưu kết quả RAG ablation tại: %s", output_path.resolve())
    return report


def run_chunk_ablation(
    data_dir: str,
    temp_dir: Path,
    test_cases: list[Any],
    output_path: Path,
) -> dict[str, Any]:
    """Chạy ablation các cấu hình chunking."""
    LOGGER.info("--- Bắt đầu Chunk-Size & Strategy Ablation ---")

    ans_cases = [c for c in test_cases if c.is_answerable]
    expected_sources = [set(c.expected_documents or c.expected_sources) for c in ans_cases]

    chunk_configs = [
        {"name": "Sliding Window (120w / 20o)", "strategy": "sliding_window", "chunk_words": 120, "overlap_words": 20},
        {"name": "Sliding Window (220w / 30o)", "strategy": "sliding_window", "chunk_words": 220, "overlap_words": 30},
        {"name": "Sliding Window (350w / 50o)", "strategy": "sliding_window", "chunk_words": 350, "overlap_words": 50},
        {"name": "Structure-Aware (Sentence Packing)", "strategy": "structure_aware", "chunk_words": 250, "overlap_words": 40},
    ]

    chunk_results: list[dict[str, Any]] = []

    for cfg in chunk_configs:
        cfg_name = cfg["name"]
        model_subdir = temp_dir / f"index_{cfg['strategy']}_{cfg['chunk_words']}"

        idx_config = IndexConfig(
            data_dir=data_dir,
            model_dir=str(model_subdir),
            chunk_words=cfg["chunk_words"],
            overlap_words=cfg["overlap_words"],
            strategy=cfg["strategy"],
            use_reranker=False,  # Để đo thuần túy tác động của chunking lên retrieval
        )
        build_index(idx_config)

        temp_retriever = Retriever(model_dir=str(model_subdir), use_reranker=False)

        ranked_docs: list[list[str]] = []
        latencies: list[float] = []

        for case in ans_cases:
            t0 = time.perf_counter()
            hits = temp_retriever.search(case.question, k=4, use_reranker=False)
            latencies.append(time.perf_counter() - t0)
            ranked_docs.append([str(h.get("source", "")) for h in hits])

        metrics = calculate_retrieval_metrics(ranked_docs, expected_sources, latencies)
        chunk_results.append(
            {
                "strategy": cfg_name,
                "total_chunks": len(temp_retriever.chunks),
                "recall_at_k": metrics["recall_at_k"],
                "hit_rate_at_1": metrics["hit_rate_at_1"],
                "mrr": metrics["mrr"],
                "ndcg_at_k": metrics["ndcg_at_k"],
                "avg_latency_ms": metrics["avg_latency_ms"],
            }
        )

    report = {
        "title": "Chunk-Size & Strategy Ablation Study",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": chunk_results,
    }
    save_json(output_path, report)
    LOGGER.info("Đã lưu kết quả Chunk ablation tại: %s", output_path.resolve())
    return report


def main() -> None:
    """Hàm chính điều phối các thực nghiệm ablation."""
    setup_logging()
    cases = load_benchmark(PROJECT_ROOT / "data/evaluation/questions.json")
    test_cases = [c for c in cases if c.split == "test"]

    # Đảm bảo index chính tồn tại
    retriever = Retriever(model_dir=PROJECT_ROOT / "models/rag_index")

    # 1. RAG Component Ablation
    rag_rep = run_rag_ablation(
        retriever,
        test_cases,
        PROJECT_ROOT / "reports/rag_ablation.json",
    )

    # 2. Chunk Ablation
    chunk_rep = run_chunk_ablation(
        data_dir="data/raw",
        temp_dir=PROJECT_ROOT / "models/ablation_scratch",
        test_cases=test_cases,
        output_path=PROJECT_ROOT / "reports/chunk_ablation.json",
    )

    print("\n" + "=" * 90)
    print(" RAG COMPONENT ABLATION REPORT")
    print("=" * 90)
    print(f"{'Component':<36} | {'Recall@K':<10} | {'Hit@1':<8} | {'MRR':<8} | {'nDCG':<8} | {'P95 Latency':<10}")
    print("-" * 90)
    for row in rag_rep["results"]:
        print(
            f"{row['component']:<36} | {row['recall_at_k']:<10.4f} | "
            f"{row['hit_rate_at_1']:<8.4f} | {row['mrr']:<8.4f} | "
            f"{row['ndcg_at_k']:<8.4f} | {row['p95_latency_ms']:<8.2f} ms"
        )
    print("=" * 90)

    print("\n" + "=" * 90)
    print(" CHUNKING STRATEGY ABLATION REPORT")
    print("=" * 90)
    print(f"{'Strategy':<36} | {'Chunks':<8} | {'Recall@K':<10} | {'MRR':<8} | {'Latency':<10}")
    print("-" * 90)
    for row in chunk_rep["results"]:
        print(
            f"{row['strategy']:<36} | {row['total_chunks']:<8} | "
            f"{row['recall_at_k']:<10.4f} | {row['mrr']:<8.4f} | "
            f"{row['avg_latency_ms']:<8.2f} ms"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
