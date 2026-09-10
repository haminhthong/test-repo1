"""Đánh giá độc lập toàn diện hệ thống RAG đa tầng (Multi-Layer Benchmark Evaluation).

Hỗ trợ đánh giá 3 tầng chất lượng:
1. Tầng 1 (Retrieval Quality): Document Recall@K, Evidence Recall@K, MRR@K, nDCG@K (chuẩn [0.0, 1.0]),
   Hit@1, Latency p50/p95. Phân rã theo từng lát cắt dữ liệu (Slices: factual, paraphrase, keyword_code, numeric, no_answer).
2. Tầng 2 (Evidence Gate & Answerability): Tỷ lệ từ chối đúng khi thiếu bằng chứng (True Abstention Rate),
   False Answer Rate, Tối ưu ngưỡng Evidence Gate tự động trên tập Dev.
3. Tầng 3 (Grounded Generation & Citation): Top Evidence Reference Token Coverage,
   Citation Reference Validity (kiểm định cú pháp [C1], [C2] và trích xuất căn thực).
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from math import nextafter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import DEFAULT_EVIDENCE_GATE_THRESHOLD
from .generation import validate_citation_references
from .ranking import tokenize
from .security import AccessContext
from .utils import save_json, setup_logging

if TYPE_CHECKING:
    from .retrieval import Retriever

LOGGER = logging.getLogger("rag_knowledge_assistant.evaluate")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data/evaluation/questions.json"
BENCHMARK_ACCESS = AccessContext(
    user_id="offline-evaluator",
    groups=("employee", "hr", "finance", "security"),
    auth_method="offline-benchmark",
)


@dataclass(frozen=True)
class BenchmarkCase:
    """Một ca kiểm thử đánh giá với nhãn ground truth phân tầng và dev/test split."""

    question: str
    expected_sources: tuple[str, ...]
    split: str
    id: str = ""
    expected_documents: tuple[str, ...] = ()
    expected_sections: tuple[str, ...] = ()
    expected_evidence: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    reference_answer: str = ""
    category: str = "factual"
    is_answerable: bool = True


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK_PATH) -> list[BenchmarkCase]:
    """Đọc và xác thực benchmark từ JSON để tránh hard-code trong mã nguồn.

    Args:
        path (Union[str, Path]): Đường dẫn tệp JSON benchmark.

    Returns:
        List[BenchmarkCase]: Danh sách các trường hợp kiểm thử đã qua xác thực.

    Raises:
        ValueError: Nếu định dạng ca kiểm thử không hợp lệ hoặc rỗng.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark phải có dạng danh sách các case.")
    cases: list[BenchmarkCase] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark case #{index} phải là object: {item!r}")
        question = str(item.get("question", "")).strip()
        raw_sources = item.get("expected_sources", [])
        if not isinstance(raw_sources, list):
            raise ValueError(f"Benchmark case #{index}: expected_sources phải là danh sách.")
        sources = tuple(str(value).strip() for value in raw_sources if str(value).strip())
        split = str(item.get("split", "test")).strip().lower()

        if not question or not sources or split not in {"dev", "test"}:
            raise ValueError(f"Benchmark case #{index} không hợp lệ: {item}")

        raw_docs = item.get("expected_documents", raw_sources)
        if not isinstance(raw_docs, list):
            raise ValueError(f"Benchmark case #{index}: expected_documents phải là danh sách.")
        docs = tuple(str(v).strip() for v in raw_docs if str(v).strip().upper() != "ABSTAIN")
        raw_answerable = item.get("is_answerable", True)
        if not isinstance(raw_answerable, bool):
            raise ValueError(f"Benchmark case #{index}: is_answerable phải là boolean.")
        is_answerable = raw_answerable
        if is_answerable and not docs:
            raise ValueError(
                f"Benchmark case #{index}: case trả lời được phải có tài liệu kỳ vọng."
            )
        if not is_answerable and docs:
            raise ValueError(
                f"Benchmark case #{index}: case không trả lời được không được có tài liệu kỳ vọng."
            )

        raw_secs = item.get("expected_sections", [])
        if not isinstance(raw_secs, list):
            raise ValueError(f"Benchmark case #{index}: expected_sections phải là danh sách.")
        sections = tuple(str(v).strip() for v in raw_secs)

        raw_evidence = item.get("expected_evidence", [])
        if not isinstance(raw_evidence, list):
            raise ValueError(f"Benchmark case #{index}: expected_evidence phải là danh sách.")
        evidence_list: list[dict[str, Any]] = []
        for ev in raw_evidence:
            if isinstance(ev, dict):
                evidence_list.append(ev)
            elif isinstance(ev, str):
                evidence_list.append({"document": ev})

        cases.append(
            BenchmarkCase(
                id=str(item.get("id", f"Q{index + 1:02d}")),
                question=question,
                expected_sources=sources,
                split=split,
                expected_documents=docs,
                expected_sections=sections,
                expected_evidence=tuple(evidence_list),
                reference_answer=str(item.get("reference_answer", "")),
                category=str(item.get("category", "factual")),
                is_answerable=is_answerable,
            )
        )

    if not cases:
        raise ValueError("Benchmark không được rỗng.")
    return cases


def unique_preserve_order(items: list[str]) -> list[str]:
    """Loại bỏ phần tử trùng lặp nhưng bảo toàn thứ tự xếp hạng ban đầu."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def calculate_retrieval_metrics(
    ranked_sources: list[list[str]],
    expected_sources: list[set[str]],
    latencies: list[float],
    *,
    deduplicate: bool = True,
    k: int | None = None,
) -> dict[str, float | int]:
    """Tính Recall@k, Hit@1, MRR, nDCG@k và phân vị latency theo chuẩn toán học [0.0, 1.0].

    Giải quyết triệt để lỗi nDCG > 1:
    1. Khi đánh giá Document Retrieval, deduplicate tài liệu nguồn trước để tránh việc
       nhiều chunk từ cùng một tài liệu liên tục cộng dồn relevance làm DCG > IDCG.
    2. Tính IDCG chuẩn xác theo số lượng tài liệu liên quan thực tế:
       ideal_k = min(len(expected), effective_k)
       idcg = sum(1 / log2(i + 2) for i in range(ideal_k))
    3. Đảm bảo bất đẳng thức toán học: 0.0 <= nDCG@k <= 1.0 dưới mọi trường hợp.
    """
    if (
        not ranked_sources
        or len(ranked_sources) != len(expected_sources)
        or len(latencies) != len(ranked_sources)
    ):
        raise ValueError("Kết quả, ground truth và latency phải cùng số mẫu, không rỗng.")
    if k is not None and k <= 0:
        raise ValueError("k phải lớn hơn 0.")

    reciprocal_ranks: list[float] = []
    ndcg_list: list[float] = []
    recall_list: list[float] = []
    hits_at_one = 0

    for sources, expected in zip(ranked_sources, expected_sources, strict=True):
        if not expected:
            continue

        eval_sources = unique_preserve_order(sources) if deduplicate else list(sources)
        effective_k = k if k is not None else len(eval_sources)
        eval_sources = eval_sources[:effective_k]
        recall_list.append(len(set(eval_sources) & expected) / len(expected))

        rank = next(
            (
                position
                for position, source in enumerate(eval_sources, start=1)
                if source in expected
            ),
            None,
        )
        hits_at_one += int(rank == 1)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        # Tính toán DCG@k
        dcg = 0.0
        for pos, source in enumerate(eval_sources, start=1):
            if source in expected:
                dcg += 1.0 / math.log2(pos + 1)

        # Tính toán IDCG@k chuẩn hóa
        ideal_relevant = min(len(expected), effective_k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))

        ndcg_val = (dcg / idcg) if idcg > 0.0 else 0.0
        # Kẹp chặt trong [0.0, 1.0] để chống sai số float
        ndcg_list.append(max(0.0, min(1.0, ndcg_val)))

    total_valid = len(reciprocal_ranks)
    if total_valid == 0:
        return {
            "recall_at_k": 1.0,
            "hit_rate_at_1": 1.0,
            "mrr": 1.0,
            "ndcg_at_k": 1.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "total_queries": 0,
        }

    latency_ms = sorted(value * 1000 for value in latencies)
    p95_index = min(len(latency_ms) - 1, max(0, int(0.95 * len(latency_ms)) - 1))
    p50_index = len(latency_ms) // 2

    return {
        "recall_at_k": round(statistics.fmean(recall_list), 4),
        "hit_rate_at_1": round(hits_at_one / total_valid, 4),
        "mrr": round(statistics.fmean(reciprocal_ranks), 4),
        "ndcg_at_k": round(statistics.fmean(ndcg_list), 4),
        "avg_latency_ms": round(statistics.fmean(latency_ms), 2),
        "p50_latency_ms": round(latency_ms[p50_index], 2),
        "p95_latency_ms": round(latency_ms[p95_index], 2),
        "total_queries": total_valid,
    }


def evaluate_evidence_retrieval(
    retrieved_chunks: list[list[dict[str, Any]]],
    cases: list[BenchmarkCase],
) -> dict[str, float]:
    """Đánh giá chất lượng truy xuất cấp độ Bằng chứng (Evidence-Level Retrieval).

    Khác với Document Retrieval chỉ so khớp tên tệp, Evidence Retrieval kiểm tra:
    - Đúng tài liệu (document/source).
    - Đúng section/chương mục (nếu case có định nghĩa).
    - Đoạn trích text có chứa chuỗi bằng chứng kỳ vọng (text_contains).
    """
    ans_cases = [c for c in cases if c.is_answerable]
    if not ans_cases:
        return {"evidence_recall_at_k": 1.0, "evidence_mrr": 1.0}

    hits_found = 0
    reciprocal_ranks: list[float] = []

    for hits, case in zip(retrieved_chunks, ans_cases, strict=True):
        if not case.expected_evidence and not case.expected_sections:
            # Fallback sang document level nếu case chưa khai báo evidence chi tiết
            doc_set = set(case.expected_documents or case.expected_sources)
            match_rank = next(
                (pos for pos, h in enumerate(hits, start=1) if h.get("source") in doc_set),
                None,
            )
        else:
            match_rank = None
            for pos, h in enumerate(hits, start=1):
                h_doc = h.get("source", "")
                h_sec = h.get("section", "")
                h_text = h.get("text", "")

                matched = False
                if case.expected_evidence:
                    for ev in case.expected_evidence:
                        ev_doc = ev.get("document", ev.get("document_id", ""))
                        ev_sec = ev.get("section")
                        ev_text = ev.get("text_contains", "")

                        doc_match = (not ev_doc) or (ev_doc in h_doc) or (h_doc in ev_doc)
                        sec_match = (not ev_sec) or (ev_sec == h_sec)
                        text_match = (not ev_text) or (ev_text.lower() in h_text.lower())

                        if doc_match and sec_match and text_match:
                            matched = True
                            break
                elif case.expected_sections:
                    doc_set = set(case.expected_documents or case.expected_sources)
                    if h_doc in doc_set and h_sec in case.expected_sections:
                        matched = True

                if matched:
                    match_rank = pos
                    break

        if match_rank is not None:
            hits_found += 1
            reciprocal_ranks.append(1.0 / match_rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(ans_cases)
    return {
        "evidence_recall_at_k": round(hits_found / total, 4) if total > 0 else 1.0,
        "evidence_mrr": round(statistics.fmean(reciprocal_ranks), 4) if reciprocal_ranks else 1.0,
    }


def evaluate_citation_metrics(
    answers: list[str], retrieved_hits: list[list[dict[str, Any]]], max_chunks: int = 4
) -> dict[str, float]:
    """Tách citation reference validity khỏi factual-sentence coverage.

    Hàm này chỉ đo contract runtime; semantic support vẫn cần human/judge
    đánh giá offline riêng.
    """
    if len(answers) != len(retrieved_hits):
        raise ValueError("answers và retrieved_hits phải cùng số mẫu.")
    if not answers:
        return {
            "citation_reference_validity": 1.0,
            "factual_sentence_citation_coverage": 1.0,
        }
    validations = [
        validate_citation_references(answer, hits, max_chunks=max_chunks)[2]
        for answer, hits in zip(answers, retrieved_hits, strict=True)
    ]
    return {
        "citation_reference_validity": round(
            statistics.fmean(float(item["citation_reference_validity"]) for item in validations),
            4,
        ),
        "factual_sentence_citation_coverage": round(
            statistics.fmean(
                float(item["factual_sentence_citation_coverage"]) for item in validations
            ),
            4,
        ),
    }


def evaluate_retrieval_comprehensive(
    retriever: Retriever,
    cases: list[BenchmarkCase],
    *,
    top_k: int = 4,
    use_reranker: bool = True,
    use_dense: bool = True,
    use_bm25: bool = True,
    access_context: AccessContext = BENCHMARK_ACCESS,
) -> dict[str, Any]:
    """Đánh giá toàn diện Retrieval chất lượng đa tầng theo từng category slice."""
    ranked_sources: list[list[str]] = []
    expected_sources: list[set[str]] = []
    latencies: list[float] = []
    retrieved_chunks_all: list[list[dict[str, Any]]] = []

    # Thống kê theo danh mục (Slice)
    category_cases: dict[str, list[BenchmarkCase]] = {}
    category_ranked: dict[str, list[list[str]]] = {}
    category_expected: dict[str, list[set[str]]] = {}
    category_latencies: dict[str, list[float]] = {}

    gate_threshold = float(
        getattr(retriever, "evidence_gate_threshold", DEFAULT_EVIDENCE_GATE_THRESHOLD)
    )
    reranker_mode = str(getattr(getattr(retriever, "reranker", None), "mode", ""))
    gate_available = use_reranker and reranker_mode == "neural"

    abstain_correct = 0
    total_unanswerable = 0
    false_answers = 0
    keyword_overlaps: list[float] = []

    for case in cases:
        started = time.perf_counter()
        results = retriever.search(
            case.question,
            k=top_k,
            use_dense=use_dense,
            use_bm25=use_bm25,
            use_reranker=use_reranker,
            access_context=access_context,
            # Không lọc theo score khi đo retrieval; raw logit có thể âm.
            min_score=float("-inf"),
        )
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        retrieved_docs = [str(item.get("source", "")) for item in results]
        ranked_sources.append(retrieved_docs)
        if case.is_answerable:
            retrieved_chunks_all.append(results)

        # Tập expected sources (bỏ qua 'ABSTAIN' trong so sánh tài liệu thật)
        exp_docs = set(case.expected_documents)
        expected_sources.append(exp_docs)

        cat = case.category
        if cat not in category_cases:
            category_cases[cat] = []
            category_ranked[cat] = []
            category_expected[cat] = []
            category_latencies[cat] = []

        category_cases[cat].append(case)
        category_ranked[cat].append(retrieved_docs)
        category_expected[cat].append(exp_docs)
        category_latencies[cat].append(elapsed)

        # Đánh giá Grounding / Abstention
        gate_all_failed = (
            not results
            or not gate_available
            or all(
                float(item.get("evidence_score", item.get("retrieval_score", float("-inf"))))
                < gate_threshold
                for item in results
            )
        )
        if not case.is_answerable:
            total_unanswerable += 1
            if gate_all_failed:
                abstain_correct += 1
            else:
                false_answers += 1
        else:
            # Top Evidence Reference Token Coverage
            if results and case.reference_answer:
                ref_tokens = set(tokenize(case.reference_answer))
                top_text_tokens = set(tokenize(results[0].get("text", "")))
                overlap = len(ref_tokens & top_text_tokens) / max(1, len(ref_tokens))
                keyword_overlaps.append(overlap)

    # Tính toán chỉ số tổng thể trên các câu hỏi có thể trả lời
    answerable_ranked = [r for r, c in zip(ranked_sources, cases, strict=True) if c.is_answerable]
    answerable_expected = [
        e for e, c in zip(expected_sources, cases, strict=True) if c.is_answerable
    ]
    answerable_latencies = [
        latency for latency, case in zip(latencies, cases, strict=True) if case.is_answerable
    ]

    base_metrics = calculate_retrieval_metrics(
        answerable_ranked, answerable_expected, answerable_latencies, k=top_k
    )
    evidence_metrics = evaluate_evidence_retrieval(retrieved_chunks_all, cases)

    # Đánh giá từng Category Slice
    slice_metrics: dict[str, Any] = {}
    for cat, cat_cases in category_cases.items():
        cat_ans_ranked = [
            r for r, c in zip(category_ranked[cat], cat_cases, strict=True) if c.is_answerable
        ]
        cat_ans_expected = [
            e for e, c in zip(category_expected[cat], cat_cases, strict=True) if c.is_answerable
        ]
        cat_ans_latencies = [
            latency
            for latency, case in zip(category_latencies[cat], cat_cases, strict=True)
            if case.is_answerable
        ]

        if cat_ans_ranked:
            slice_metrics[cat] = calculate_retrieval_metrics(
                cat_ans_ranked, cat_ans_expected, cat_ans_latencies, k=top_k
            )
        else:
            slice_metrics[cat] = {
                "total_queries": len(cat_cases),
                "is_unanswerable_slice": True,
            }

    true_abstain_rate = (
        round(abstain_correct / total_unanswerable, 4) if total_unanswerable > 0 else 1.0
    )
    false_answer_rate = (
        round(false_answers / total_unanswerable, 4) if total_unanswerable > 0 else 0.0
    )
    avg_keyword_coverage = round(statistics.fmean(keyword_overlaps), 4) if keyword_overlaps else 0.0

    return {
        **base_metrics,
        **evidence_metrics,
        "true_abstain_rate": true_abstain_rate,
        "false_answer_rate": false_answer_rate,
        "unanswerable_evaluated": total_unanswerable,
        "top_evidence_token_coverage": avg_keyword_coverage,
        "avg_keyword_coverage": avg_keyword_coverage,  # Bí danh tương thích ngược
        "slices": slice_metrics,
    }


def tune_evidence_gate_threshold(
    dev_cases: list[BenchmarkCase],
    retriever: Retriever,
    *,
    target_false_answer_rate: float = 0.05,
    top_k: int = 4,
    use_reranker: bool = True,
) -> float:
    """Tối ưu ngưỡng Evidence Gate tự động CHỈ trên tập Dev (không rò rỉ dữ liệu Test).

    Mục tiêu: Maximize Answerable Recall subject to False Answer Rate <= target_false_answer_rate.

    Args:
        dev_cases: Danh sách ca kiểm thử trên split='dev'.
        retriever: Thể hiện Retriever đang đánh giá.
        target_false_answer_rate: Ngưỡng tối đa chấp nhận việc trả lời nhầm trên câu unanswerable.
        top_k: Số lượng chunk ứng viên.
        use_reranker: Cờ reranker.

    Returns:
        float: Ngưỡng điểm evidence_score được chọn.
    """
    if not 0.0 <= target_false_answer_rate <= 1.0:
        raise ValueError("target_false_answer_rate phải nằm trong khoảng [0.0, 1.0].")
    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0.")
    assert all(c.split == "dev" for c in dev_cases), (
        "LỖI NGUY HIỂM: Chỉ được tune threshold trên tập DEV!"
    )

    unans_cases = [c for c in dev_cases if not c.is_answerable]
    ans_cases = [c for c in dev_cases if c.is_answerable]

    if not unans_cases or not ans_cases:
        return float(getattr(retriever, "evidence_gate_threshold", DEFAULT_EVIDENCE_GATE_THRESHOLD))

    # Thu thập điểm số evidence cao nhất cho mỗi case
    unans_scores: list[float] = []
    for c in unans_cases:
        hits = retriever.search(
            c.question,
            k=top_k,
            use_reranker=use_reranker,
            min_score=float("-inf"),
            access_context=BENCHMARK_ACCESS,
        )
        top_sc = max(
            (float(h.get("evidence_score", h.get("retrieval_score", float("-inf")))) for h in hits),
            default=float("-inf"),
        )
        unans_scores.append(top_sc)

    ans_scores: list[float] = []
    for c in ans_cases:
        hits = retriever.search(
            c.question,
            k=top_k,
            use_reranker=use_reranker,
            min_score=float("-inf"),
            access_context=BENCHMARK_ACCESS,
        )
        top_sc = max(
            (float(h.get("evidence_score", h.get("retrieval_score", float("-inf")))) for h in hits),
            default=float("-inf"),
        )
        ans_scores.append(top_sc)

    best_threshold = float(
        getattr(retriever, "evidence_gate_threshold", DEFAULT_EVIDENCE_GATE_THRESHOLD)
    )
    best_ans_recall = -1.0
    min_false_rate = float("inf")
    feasible_found = False

    # Chỉ thử tại các score quan sát được và ngay phía trên score đó. Cách này
    # giữ đúng thang raw logit, không ép score về khoảng xác suất [0, 1].
    observed_scores = sorted(
        {score for score in unans_scores + ans_scores + [best_threshold] if math.isfinite(score)}
    )
    thresholds = sorted(
        set(observed_scores) | {nextafter(score, float("inf")) for score in observed_scores}
    )
    for tau in thresholds:
        false_ans_cnt = sum(sc >= tau for sc in unans_scores)
        far = false_ans_cnt / len(unans_scores)

        true_ans_cnt = sum(sc >= tau for sc in ans_scores)
        ans_recall = true_ans_cnt / len(ans_scores)

        if far <= target_false_answer_rate:
            if (
                not feasible_found
                or ans_recall > best_ans_recall
                or (ans_recall == best_ans_recall and far < min_false_rate)
            ):
                feasible_found = True
                best_ans_recall = ans_recall
                best_threshold = tau
                min_false_rate = far
        elif not feasible_found and far < min_false_rate:
            # Nếu không có ngưỡng đạt target, chọn ngưỡng có FAR thấp nhất.
            best_threshold = tau
            min_false_rate = far

    if best_ans_recall < 0:
        best_ans_recall = sum(score >= best_threshold for score in ans_scores) / len(ans_scores)

    LOGGER.info(
        "Dev Gate Tuning hoàn tất: chọn threshold=%.2f (Dev Answerable Recall=%.2f%%, FAR=%.2f%%)",
        best_threshold,
        best_ans_recall * 100,
        min_false_rate * 100,
    )
    return best_threshold


def run_evaluation(
    model_dir: str | Path | None = None,
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    output_report_path: str | Path | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    """Tune/so sánh trên Dev, sau đó chạy canonical đúng một lần trên Test."""
    setup_logging()
    from .retrieval import Retriever

    cases = load_benchmark(benchmark_path)
    dev_cases = [case for case in cases if case.split == "dev"]
    test_cases = [case for case in cases if case.split == "test"]
    if not dev_cases or not test_cases:
        raise ValueError("Benchmark phải có cả split dev và test.")

    retriever = Retriever(model_dir=model_dir, use_reranker=True)
    calibrated_threshold = tune_evidence_gate_threshold(
        dev_cases, retriever, target_false_answer_rate=0.05, top_k=top_k
    )
    retriever.evidence_gate_threshold = calibrated_threshold

    dev_pipelines = {
        "bm25_only": evaluate_retrieval_comprehensive(
            retriever, dev_cases, top_k=top_k, use_dense=False, use_bm25=True, use_reranker=False
        ),
        "dense_only": evaluate_retrieval_comprehensive(
            retriever, dev_cases, top_k=top_k, use_dense=True, use_bm25=False, use_reranker=False
        ),
        "hybrid_rrf": evaluate_retrieval_comprehensive(
            retriever, dev_cases, top_k=top_k, use_reranker=False
        ),
        "hybrid_rrf_multilingual_reranker": evaluate_retrieval_comprehensive(
            retriever, dev_cases, top_k=top_k, use_reranker=True
        ),
    }
    final_metrics = evaluate_retrieval_comprehensive(
        retriever, test_cases, top_k=top_k, use_reranker=True
    )
    policy = {
        "dense_k": int(retriever.config.get("candidate_pool_k", retriever.default_candidate_k)),
        "sparse_k": int(retriever.config.get("candidate_pool_k", retriever.default_candidate_k)),
        "rrf_k": retriever.rrf_k,
        "rerank_top_k": retriever.rerank_top_k,
        "context_k": top_k,
        "reranker_model": retriever.config.get("reranker_model"),
        "evidence_gate_threshold": calibrated_threshold,
        "evaluation_dataset": Path(benchmark_path).as_posix(),
    }
    dev_report = {
        "selection_split": "dev",
        "evaluation_split": "test",
        "retrieval_config": policy,
        "pipelines": dev_pipelines,
    }
    final_report = {
        "system_name": "Vietnamese Policy RAG",
        "selection_split": "dev",
        "evaluation_split": "test",
        "test_threshold_selected_on_dev": True,
        "total_test_cases": len(test_cases),
        "retrieval_config": policy,
        "canonical_pipeline": "Dense + BM25 + RRF + Cross-Encoder + Evidence Gate",
        "metrics": final_metrics,
        "limitations": [
            "Benchmark được đọc từ file đã chọn; cần freeze dữ liệu trước khi so sánh các lần chạy.",
            "Runtime citation guard kiểm tra citation ID và sentence coverage, không tự chứng minh entailment.",
            "PDF scan/OCR và bảng PDF phức tạp chưa thuộc phạm vi V1.",
        ],
    }
    save_json(PROJECT_ROOT / "reports/dev_metrics.json", dev_report)
    save_json(PROJECT_ROOT / "reports/final_test_metrics.json", final_report)
    if output_report_path:
        save_json(output_report_path, final_report)
    return {"dev": dev_report, "final": final_report}


if __name__ == "__main__":
    run_evaluation()
