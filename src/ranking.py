"""Module xếp hạng lai (Hybrid Ranking & Fusion Engine).

Tệp này thực hiện các thuật toán:
1. Tách từ (Tokenization) tiếng Việt có dấu.
2. Bộ chỉ mục từ khóa thực thụ (BM25 Index) dựa trên thuật toán Robertson BM25Okapi.
3. Thuật toán dung hợp thứ hạng Reciprocal Rank Fusion (RRF).
4. Mô hình xếp hạng lại sâu Cross-Encoder Reranker với cơ chế dự phòng (Fallback).
5. Các hàm bổ trợ tương thích ngược (lexical_overlap, bm25_score_single, hybrid_score).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

LOGGER = logging.getLogger("rag_knowledge_assistant.ranking")

# Regex nhận diện các từ unicode (hỗ trợ tiếng Việt đầy đủ)
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tách văn bản thành danh sách từ (token) dạng chữ thường (case-folded).

    Args:
        text (str): Chuỗi văn bản đầu vào.

    Returns:
        List[str]: Danh sách các token thu được.
    """
    return TOKEN_PATTERN.findall(text.casefold())


def get_token_set(text: str) -> set[str]:
    """Tách văn bản và chuyển thành tập hợp các từ độc nhất.

    Args:
        text (str): Chuỗi văn bản đầu vào.

    Returns:
        Set[str]: Tập hợp các token độc nhất.
    """
    return set(tokenize(text))


def lexical_overlap(query: str, document: str) -> float:
    """Tính tỷ lệ trùng lặp từ khóa (Jaccard-like Overlap) giữa câu hỏi và văn bản.

    Args:
        query (str): Câu hỏi/truy vấn của người dùng.
        document (str): Đoạn văn bản tài liệu.

    Returns:
        float: Giá trị trong khoảng [0.0, 1.0] thể hiện tỷ lệ token của query có trong document.
    """
    query_tokens = get_token_set(query)
    if not query_tokens:
        return 0.0
    doc_tokens = get_token_set(document)
    intersection = query_tokens & doc_tokens
    return len(intersection) / len(query_tokens)


def bm25_score_single(
    query_tokens: list[str],
    doc_tokens: list[str],
    total_docs: int = 100,
    doc_freqs: dict[str, int] | None = None,
    avg_doc_len: float = 200.0,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Tính điểm BM25 cho một văn bản đối với danh sách token truy vấn.

    Args:
        query_tokens (List[str]): Danh sách các từ trong câu hỏi.
        doc_tokens (List[str]): Danh sách các từ trong văn bản.
        total_docs (int): Tổng số văn bản trong corpus.
        doc_freqs (Dict[str, int], optional): Số văn bản chứa từng từ.
        avg_doc_len (float): Độ dài trung bình của văn bản trong corpus.
        k1 (float): Term Frequency saturation parameter.
        b (float): Length Normalization parameter.

    Returns:
        float: Điểm số BM25 (chưa chuẩn hóa).
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_len = len(doc_tokens)
    score = 0.0
    doc_freqs = doc_freqs or {}

    tf_map: dict[str, int] = {}
    for term in doc_tokens:
        tf_map[term] = tf_map.get(term, 0) + 1

    for term in set(query_tokens):
        if term not in tf_map:
            continue

        freq = tf_map[term]
        df = doc_freqs.get(term, 1)
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        numerator = freq * (k1 + 1.0)
        denominator = freq + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len)))
        score += idf * (numerator / denominator)

    return max(0.0, score)


def hybrid_score(
    dense_score: float,
    lexical_score: float,
    dense_weight: float = 0.85,
) -> float:
    """Kết hợp điểm Dense Cosine Similarity và điểm Lexical Relevance thành điểm tổng hợp tuyến tính.

    Công thức: Combine = dense_weight * Normal(dense_score) + (1 - dense_weight) * lexical_score

    Args:
        dense_score (float): Điểm tương đồng Cosine từ FAISS index [-1.0, 1.0].
        lexical_score (float): Điểm trùng lặp từ khóa [0.0, 1.0].
        dense_weight (float): Tỷ trọng dành cho Dense Retrieval (từ 0.0 đến 1.0, mặc định: 0.85).

    Returns:
        float: Điểm số tổng hợp chuẩn hóa trong khoảng [0.0, 1.0].

    Raises:
        ValueError: Nếu dense_weight không nằm trong khoảng [0.0, 1.0].
    """
    if not (0.0 <= dense_weight <= 1.0):
        raise ValueError(
            f"dense_weight={dense_weight} phải nằm trong khoảng [0.0, 1.0]"
        )

    normalized_dense = min(max((dense_score + 1.0) / 2.0, 0.0), 1.0)
    final_score = dense_weight * normalized_dense + (1.0 - dense_weight) * lexical_score
    return round(final_score, 4)


class BM25Index:
    """Chỉ mục từ khóa BM25Okapi độc lập phục vụ tìm kiếm Lexical Top-N.

    Attributes:
        tokenized_corpus (List[List[str]]): Danh sách các token của từng chunk trong corpus.
        bm25 (BM25Okapi): Thể hiện của thuật toán BM25Okapi.
    """

    def __init__(self, tokenized_corpus: list[list[str]] | None = None) -> None:
        """Khởi tạo chỉ mục BM25 từ kho ngữ liệu đã tokenize."""
        self.tokenized_corpus: list[list[str]] = tokenized_corpus or []
        self._bm25_model: Any = None
        if self.tokenized_corpus:
            self._build()

    def _build(self) -> None:
        """Xây dựng mô hình BM25Okapi và cache document frequency cho fallback nội bộ."""
        try:
            from rank_bm25 import BM25Okapi

            self._bm25_model = BM25Okapi(self.tokenized_corpus)
        except ImportError:
            LOGGER.warning("rank_bm25 chưa được cài đặt, sử dụng fallback BM25 nội bộ.")
            self._bm25_model = None

        # Cache document frequency & average doc length để tính IDF chính xác
        # cho cả nhánh fallback nội bộ.
        df: dict[str, int] = {}
        total_len = 0
        for tokens in self.tokenized_corpus:
            total_len += len(tokens)
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._doc_freqs: dict[str, int] = df
        self._avg_doc_len: float = (
            total_len / len(self.tokenized_corpus)
            if self.tokenized_corpus
            else 0.0
        )

    @classmethod
    def from_texts(cls, texts: list[str]) -> BM25Index:
        """Tạo BM25Index từ danh sách chuỗi văn bản."""
        tokenized = [tokenize(text) for text in texts]
        return cls(tokenized)

    def search(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        """Tìm kiếm top_k đoạn văn bản khớp từ khóa BM25 tốt nhất cho câu hỏi.

        Args:
            query (str): Câu hỏi của người dùng.
            top_k (int): Số lượng kết quả ứng viên cần lấy (mặc định: 30).

        Returns:
            List[Tuple[int, float]]: Danh sách (vị trí chunk trong corpus, điểm BM25).
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self.tokenized_corpus:
            return []

        if self._bm25_model is not None:
            scores = self._bm25_model.get_scores(query_tokens)
        else:
            # Fallback nếu không có rank_bm25: dùng bm25_score_single với
            # total_docs và doc_freqs lấy từ corpus đã cache.
            scores = [
                bm25_score_single(
                    query_tokens,
                    doc_tokens,
                    total_docs=len(self.tokenized_corpus),
                    doc_freqs=self._doc_freqs,
                    avg_doc_len=self._avg_doc_len,
                )
                for doc_tokens in self.tokenized_corpus
            ]

        # Sắp xếp và lấy top-k có điểm > 0
        scored_pairs = [
            (idx, float(score)) for idx, score in enumerate(scores) if score > 0.0
        ]
        scored_pairs.sort(key=lambda item: item[1], reverse=True)
        return scored_pairs[:top_k]

    def to_dict(self) -> dict[str, Any]:
        """Tuần tự hóa dữ liệu BM25 sang dạng dictionary để lưu JSON."""
        return {
            "version": "1.0",
            "corpus_size": len(self.tokenized_corpus),
            "tokenized_corpus": self.tokenized_corpus,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        """Khôi phục BM25Index từ dictionary đã lưu."""
        tokens = data.get("tokenized_corpus", [])
        return cls(tokens)


def reciprocal_rank_fusion(
    dense_ranks: dict[int, int],
    bm25_ranks: dict[int, int],
    k: int = 60,
) -> dict[int, float]:
    """Dung hợp thứ hạng ứng viên từ Dense và BM25 bằng Reciprocal Rank Fusion (RRF).

    Công thức:
        RRF(d) = sum(1 / (k + rank_i(d)))
    trong đó rank là chỉ số thứ tự 1-indexed (1, 2, 3, ...).

    Ưu điểm:
    - Không bị ảnh hưởng bởi thang đo điểm số khác biệt giữa Cosine [-1, 1] và BM25 [0, inf).
    - Cân bằng tự nhiên giữa tín hiệu ngữ nghĩa (Dense) và từ khóa chính xác (BM25).

    Args:
        dense_ranks (Dict[int, int]): Ánh xạ chunk_idx -> thứ hạng từ Dense Search (1-indexed).
        bm25_ranks (Dict[int, int]): Ánh xạ chunk_idx -> thứ hạng từ BM25 Search (1-indexed).
        k (int): Hằng số RRF làm mượt (mặc định: 60).

    Returns:
        Dict[int, float]: Ánh xạ chunk_idx -> điểm RRF đã tính toán.
    """
    all_candidate_indices = set(dense_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores: dict[int, float] = {}

    for doc_idx in all_candidate_indices:
        score = 0.0
        if doc_idx in dense_ranks:
            score += 1.0 / (k + dense_ranks[doc_idx])
        if doc_idx in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[doc_idx])
        rrf_scores[doc_idx] = score

    return rrf_scores


from dataclasses import asdict, dataclass


@dataclass
class RetrievalCandidate:
    """Cấu trúc dữ liệu đại diện cho một ứng viên truy xuất với hệ thống điểm phân tầng."""

    chunk_id: str
    document_id: str
    source: str
    page: int | None = None
    section: str | None = None
    text: str = ""
    dense_score: float = 0.0
    dense_rank: int | None = None
    bm25_score: float = 0.0
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    reranker_score: float | None = None
    evidence_score: float = 0.0
    gate_passed: bool = False
    source_path: str = ""
    security_scope: tuple[str, ...] = ("public",)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi thành từ điển JSON-serializable."""
        data = asdict(self)
        data["score"] = self.evidence_score
        data["retrieval_score"] = self.evidence_score
        return data


class CrossEncoderReranker:
    """Mô hình xếp hạng lại sâu (Cross-Encoder Reranker) chấm điểm cặp (query, chunk).

    Nhận vào Top Candidates từ RRF và tính điểm liên quan ngữ nghĩa chi tiết ở cấp độ token tương tác chéo.
    Tích hợp cơ chế Fallback nếu không có GPU/tải mô hình thất bại.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = True,
    ) -> None:
        """Khởi tạo Cross-Encoder Reranker."""
        self.model_name = model_name
        self.enabled = enabled
        self._model: Any = None
        self._init_attempted = False

    @property
    def mode(self) -> str:
        """Trạng thái hoạt động thực tế của reranker: 'neural', 'fallback', hoặc 'disabled'."""
        if not self.enabled:
            return "disabled"
        model = self._get_model()
        return "neural" if model is not None else "fallback"

    def _get_model(self) -> Any:
        """Tải mô hình CrossEncoder theo cơ chế Lazy Loading."""
        if not self.enabled:
            return None
        if not self._init_attempted:
            self._init_attempted = True
            try:
                from sentence_transformers import CrossEncoder

                LOGGER.info("Đang tải Cross-Encoder Reranker: %s", self.model_name)
                self._model = CrossEncoder(self.model_name)
                LOGGER.info("Cross-Encoder Reranker sẵn sàng.")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Không thể khởi tạo Cross-Encoder '%s' (%s). Kích hoạt cơ chế Fallback Ranking.",
                    self.model_name,
                    exc,
                )
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Xếp hạng lại danh sách ứng viên và trả về top_k tốt nhất kèm điểm rerank.

        Args:
            query (str): Câu hỏi của người dùng.
            candidates (List[Dict[str, Any]]): Danh sách các chunk ứng viên đã qua RRF.
            top_k (int): Số lượng kết quả giữ lại (mặc định: 5).

        Returns:
            List[Dict[str, Any]]: Danh sách top_k chunk đã được xếp hạng lại theo evidence_score giảm dần.
        """
        if not candidates:
            return []

        model = self._get_model()

        if model is not None:
            try:
                pairs = [(query, str(cand.get("text", ""))) for cand in candidates]
                raw_scores = model.predict(pairs)

                # Chuẩn hóa logit qua hàm Sigmoid: 1 / (1 + exp(-x)) thành điểm tương quan [0.0, 1.0].
                # Lưu ý: Đây là transformed cross-attention relevance score, không phải xác suất đã calibrate.
                normalized_scores = [
                    1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores
                ]

                for cand, score in zip(candidates, normalized_scores, strict=True):
                    sc = round(score, 4)
                    cand["reranker_score"] = sc
                    cand["evidence_score"] = sc
                    cand["rerank_score"] = sc
                    cand["retrieval_score"] = sc

                candidates.sort(key=lambda item: item["evidence_score"], reverse=True)
                return candidates[:top_k]
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Lỗi trong quá trình suy luận Cross-Encoder: %s", exc)

        # Fallback Scorer: Kết hợp RRF score và Lexical Overlap
        for cand in candidates:
            rrf_sc = float(cand.get("rrf_score", 0.0))
            overlap_sc = lexical_overlap(query, str(cand.get("text", "")))
            fallback_score = min(max(rrf_sc * 25.0 * 0.7 + overlap_sc * 0.3, 0.0), 1.0)
            sc = round(fallback_score, 4)
            cand["reranker_score"] = None  # Không gán neural score giả
            cand["evidence_score"] = sc
            cand["rerank_score"] = sc
            cand["retrieval_score"] = sc

        candidates.sort(key=lambda item: item["evidence_score"], reverse=True)
        return candidates[:top_k]

