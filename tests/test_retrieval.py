"""Unit tests cho module ranking, BM25Index, RRF, Cross-Encoder và nDCG metric guarantees."""

from __future__ import annotations

import pytest

from src.evaluate import (
    BenchmarkCase,
    calculate_retrieval_metrics,
    evaluate_retrieval_comprehensive,
    load_benchmark,
    tune_evidence_gate_threshold,
)
from src.ranking import (
    BM25Index,
    CrossEncoderReranker,
    bm25_score_single,
    hybrid_score,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_vietnamese_text():
    """Kiểm tra việc tách từ tiếng Việt có dấu."""
    text = "Nhân viên có 12 ngày phép năm!"
    tokens = tokenize(text)
    assert "nhân" in tokens
    assert "viên" in tokens
    assert "12" in tokens
    assert "phép" in tokens


def test_bm25_score_calculation():
    """Kiểm tra tính toán điểm BM25 cho câu chứa từ khóa."""
    query_tokens = ["bảo", "mật"]
    doc_tokens_relevant = ["hướng", "dẫn", "bảo", "mật", "thông", "tin"]
    doc_tokens_irrelevant = ["chính", "sách", "nghỉ", "phép", "năm"]

    score_rel = bm25_score_single(query_tokens, doc_tokens_relevant)
    score_irrel = bm25_score_single(query_tokens, doc_tokens_irrelevant)

    assert score_rel > score_irrel
    assert score_irrel == 0.0


def test_bm25_index_corpus_search():
    """Kiểm tra BM25Index tìm kiếm chính xác từ khóa trong corpus."""
    docs = [
        "Chính sách nghỉ phép nhân viên 12 ngày một năm",
        "Quy định bật xác thực 2FA cho mọi tài khoản công ty",
        "Hóa đơn đỏ VAT là bắt buộc khi hoàn ứng chi phí công tác",
    ]
    bm25 = BM25Index.from_texts(docs)

    # Tìm kiếm từ khóa chính xác "VAT"
    hits_vat = bm25.search("hóa đơn VAT", top_k=2)
    assert len(hits_vat) > 0
    assert hits_vat[0][0] == 2  # Doc index 2 phải đứng đầu

    # Tìm kiếm "2FA"
    hits_2fa = bm25.search("xác thực 2FA", top_k=2)
    assert len(hits_2fa) > 0
    assert hits_2fa[0][0] == 1


def test_reciprocal_rank_fusion_math():
    """Kiểm tra tính toán điểm Reciprocal Rank Fusion (RRF)."""
    dense_ranks = {0: 1, 1: 2}
    bm25_ranks = {0: 1, 2: 1}

    rrf_scores = reciprocal_rank_fusion(dense_ranks, bm25_ranks, k=60)
    assert rrf_scores[0] > rrf_scores[2] > rrf_scores[1]


def test_cross_encoder_reranker_fallback_scoring():
    """Kiểm tra CrossEncoderReranker hoạt động trơn tru ở chế độ fallback."""
    reranker = CrossEncoderReranker(enabled=False)
    assert reranker.mode == "disabled"

    candidates = [
        {"text": "Chính sách nghỉ phép 12 ngày", "rrf_score": 0.03},
        {"text": "Bảo mật thông tin và mật khẩu", "rrf_score": 0.01},
    ]
    reranked = reranker.rerank("nghỉ phép", candidates, top_k=2)

    assert len(reranked) == 2
    assert "rerank_score" in reranked[0]
    assert "evidence_score" in reranked[0]
    assert reranked[0]["evidence_score"] >= reranked[1]["evidence_score"]
    assert reranked[0]["text"].startswith("Chính sách nghỉ phép")


def test_hybrid_score_boundaries():
    """Kiểm tra biên của điểm số hybrid_score."""
    with pytest.raises(ValueError, match="dense_weight"):
        hybrid_score(0.8, 0.5, dense_weight=1.5)

    with pytest.raises(ValueError, match="dense_weight"):
        hybrid_score(0.8, 0.5, dense_weight=-0.1)

    assert hybrid_score(1.0, 1.0, dense_weight=0.8) == 1.0


def test_benchmark_contains_independent_dev_and_test_splits():
    """Benchmark phải tách tập chọn tham số khỏi tập báo cáo cuối."""
    cases = load_benchmark()
    assert {case.split for case in cases} == {"dev", "test"}
    assert all(case.question and case.expected_sources for case in cases)


def test_retrieval_metrics_use_first_relevant_rank():
    """MRR sử dụng đúng vị trí đầu tiên chứa một trong các nguồn hợp lệ."""
    metrics = calculate_retrieval_metrics(
        [["wrong.txt", "right.txt"], ["right.txt"]],
        [{"right.txt"}, {"right.txt"}],
        [0.01, 0.02],
    )
    assert metrics["recall_at_k"] == 1.0
    assert metrics["hit_rate_at_1"] == 0.5
    assert metrics["mrr"] == 0.75
    assert 0.0 <= metrics["ndcg_at_k"] <= 1.0


def test_recall_at_k_measures_relevant_document_coverage():
    """Recall@K phải phản ánh tỷ lệ tài liệu đúng được tìm thấy."""
    metrics = calculate_retrieval_metrics(
        [["right.txt"]],
        [{"right.txt", "also_right.txt"}],
        [0.01],
        k=2,
    )

    assert metrics["recall_at_k"] == 0.5


def test_ndcg_is_always_between_zero_and_one():
    """Bắt buộc nDCG luôn nằm trong khoảng [0.0, 1.0], không thể vượt quá 1.0."""
    # Kịch bản nguy hiểm: Nhiều chunk cùng thuộc 1 document xuất hiện ở top-k
    ranked_chunks = [
        ["policy_leave.txt", "policy_leave.txt", "policy_leave.txt", "other.txt"],
        ["security.txt", "security.txt"],
        ["wrong1.txt", "wrong2.txt"],
    ]
    expected = [
        {"policy_leave.txt"},
        {"security.txt"},
        {"right.txt"},
    ]
    latencies = [0.01, 0.02, 0.03]

    metrics = calculate_retrieval_metrics(ranked_chunks, expected, latencies, k=4)
    ndcg = float(metrics["ndcg_at_k"])

    assert 0.0 <= ndcg <= 1.0, f"LỖI TOÁN HỌC: nDCG={ndcg} vượt ra ngoài [0.0, 1.0]!"


def test_ndcg_uses_requested_k_for_ideal_gain():
    """nDCG phải phạt trường hợp chỉ tìm thấy một phần tài liệu liên quan."""
    metrics = calculate_retrieval_metrics(
        [["right.txt"]],
        [{"right.txt", "also_right.txt"}],
        [0.01],
        k=2,
    )

    assert metrics["ndcg_at_k"] < 1.0


def test_document_relevance_not_counted_twice():
    """Đảm bảo tài liệu trùng lặp trong top-k không bị cộng dồn relevance làm sai lệch DCG."""
    # Nếu không deduplicate, 3 chunk cùng tài liệu đúng sẽ cộng 1 + 1/log2(3) + 1/log2(4) > 2
    # Với IDCG = 1, nếu tính sai nDCG sẽ > 2.
    metrics = calculate_retrieval_metrics(
        [["doc1.txt", "doc1.txt", "doc1.txt"]],
        [{"doc1.txt"}],
        [0.01],
        k=4,
    )
    assert metrics["ndcg_at_k"] == 1.0


def test_evidence_gate_threshold_selected_on_dev_only():
    """Hàm tune_evidence_gate_threshold chỉ cho phép chạy trên tập DEV."""

    class FakeRetriever:
        evidence_gate_threshold = 0.25

        def search(self, *args, **kwargs):
            return [{"evidence_score": 0.8, "retrieval_score": 0.8}]

    dev_cases = [
        BenchmarkCase(
            question="Q1", expected_sources=("doc.txt",), split="dev", is_answerable=True
        ),
        BenchmarkCase(
            question="Q2", expected_sources=("ABSTAIN",), split="dev", is_answerable=False
        ),
    ]
    tau = tune_evidence_gate_threshold(dev_cases, FakeRetriever())
    assert 0.1 <= tau <= 0.9

    test_case = BenchmarkCase(
        question="Q3", expected_sources=("doc.txt",), split="test", is_answerable=True
    )
    with pytest.raises(AssertionError, match="DEV"):
        tune_evidence_gate_threshold([test_case], FakeRetriever())


def test_evidence_gate_tuning_preserves_raw_logit_scale():
    """Threshold Dev phải theo raw logit, không bị khóa trong khoảng [0, 1]."""

    class RawLogitRetriever:
        evidence_gate_threshold = 1.72

        def search(self, question, *args, **kwargs):
            score = 2.4 if question == "answerable" else 1.1
            return [{"evidence_score": score, "retrieval_score": score}]

    cases = [
        BenchmarkCase(
            question="answerable", expected_sources=("doc.txt",), split="dev", is_answerable=True
        ),
        BenchmarkCase(
            question="unanswerable", expected_sources=("ABSTAIN",), split="dev", is_answerable=False
        ),
    ]
    threshold = tune_evidence_gate_threshold(cases, RawLogitRetriever())
    assert threshold > 1.1


def test_evaluation_keeps_negative_reranker_logits() -> None:
    """Benchmark không được loại ứng viên chỉ vì raw logit nhỏ hơn 0."""

    class FakeReranker:
        mode = "neural"

    class FakeRetriever:
        evidence_gate_threshold = 1.72
        reranker = FakeReranker()

        def search(self, *args, **kwargs):
            assert kwargs["min_score"] == float("-inf")
            return [
                {
                    "source": "policy.txt",
                    "text": "Nội dung chính sách",
                    "evidence_score": -0.4,
                    "retrieval_score": -0.4,
                }
            ]

    case = BenchmarkCase(
        question="Câu hỏi",
        expected_sources=("policy.txt",),
        expected_documents=("policy.txt",),
        split="test",
        reference_answer="Nội dung chính sách",
    )
    metrics = evaluate_retrieval_comprehensive(FakeRetriever(), [case], top_k=4)

    assert metrics["recall_at_k"] == 1.0
