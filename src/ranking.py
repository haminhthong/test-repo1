"""BM25, RRF và multilingual Cross-Encoder cho hybrid retrieval."""

from __future__ import annotations

import logging
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
        """Xây dựng BM25Okapi; thiếu dependency là lỗi cấu hình rõ ràng."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("Cần cài rank-bm25 để sử dụng BM25 retrieval.") from exc
        self._bm25_model = BM25Okapi(self.tokenized_corpus)

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
        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")
        query_tokens = tokenize(query)
        if not query_tokens or not self.tokenized_corpus:
            return []

        scores = self._bm25_model.get_scores(query_tokens)

        # Sắp xếp và lấy top-k có điểm > 0
        scored_pairs = [(idx, float(score)) for idx, score in enumerate(scores) if score > 0.0]
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
    dense_ranks: dict[Any, int],
    bm25_ranks: dict[Any, int],
    k: int = 60,
) -> dict[Any, float]:
    """Dung hợp thứ hạng ứng viên từ Dense và BM25 bằng Reciprocal Rank Fusion (RRF).

    Công thức:
        RRF(d) = sum(1 / (k + rank_i(d)))
    trong đó rank là chỉ số thứ tự 1-indexed (1, 2, 3, ...).

    Ưu điểm:
    - Không bị ảnh hưởng bởi thang đo điểm số khác biệt giữa Cosine [-1, 1] và BM25 [0, inf).
    - Cân bằng tự nhiên giữa tín hiệu ngữ nghĩa (Dense) và từ khóa chính xác (BM25).

    Args:
        dense_ranks (Dict[Any, int]): Ánh xạ candidate ID -> thứ hạng Dense (1-indexed).
        bm25_ranks (Dict[Any, int]): Ánh xạ candidate ID -> thứ hạng BM25 (1-indexed).
        k (int): Hằng số RRF làm mượt (mặc định: 60).

    Returns:
        Dict[Any, float]: Ánh xạ candidate ID -> điểm RRF đã tính toán.
    """
    if k <= 0:
        raise ValueError("RRF k phải lớn hơn 0.")
    if any(rank <= 0 for rank in [*dense_ranks.values(), *bm25_ranks.values()]):
        raise ValueError("RRF rank phải bắt đầu từ 1.")

    all_candidate_indices = set(dense_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores: dict[Any, float] = {}

    for doc_idx in all_candidate_indices:
        score = 0.0
        if doc_idx in dense_ranks:
            score += 1.0 / (k + dense_ranks[doc_idx])
        if doc_idx in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[doc_idx])
        rrf_scores[doc_idx] = score

    return rrf_scores


class CrossEncoderReranker:
    """Mô hình xếp hạng lại sâu (Cross-Encoder Reranker) chấm điểm cặp (query, chunk).

    Nhận vào Top Candidates từ RRF và tính điểm liên quan ngữ nghĩa chi tiết ở cấp độ token tương tác chéo.
    Tích hợp cơ chế Fallback nếu không có GPU/tải mô hình thất bại.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        enabled: bool = True,
    ) -> None:
        """Khởi tạo Cross-Encoder Reranker."""
        self.model_name = model_name
        self.enabled = enabled
        self._model: Any = None
        self._init_attempted = False

    @property
    def mode(self) -> str:
        """Trạng thái thực tế: ``neural``, ``unavailable`` hoặc ``disabled``."""
        if not self.enabled:
            return "disabled"
        model = self._get_model()
        return "neural" if model is not None else "unavailable"

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
                    "Không thể khởi tạo Cross-Encoder '%s' (%s). Chuyển sang evidence-only.",
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

                # Giữ raw logit. Sigmoid không biến điểm thành xác suất đã
                # calibrate và làm sai semantics của evidence threshold.
                for cand, raw_score in zip(candidates, raw_scores, strict=True):
                    sc = round(float(raw_score), 4)
                    cand["reranker_score"] = sc
                    cand["reranker_logit"] = sc
                    cand["evidence_score"] = sc
                    cand["rerank_score"] = sc
                    cand["retrieval_score"] = sc

                candidates.sort(key=lambda item: item["evidence_score"], reverse=True)
                return candidates[:top_k]
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Lỗi trong quá trình suy luận Cross-Encoder: %s", exc)

        # Không tự tạo điểm neural giả. Giữ RRF để debug/baseline, còn service
        # sẽ chuyển sang sources_only vì evidence gate cần raw logit của model.
        for cand in candidates:
            rrf_sc = float(cand.get("rrf_score", 0.0))
            cand["reranker_score"] = None  # Không gán neural score giả
            cand["reranker_logit"] = None
            cand["evidence_score"] = rrf_sc
            cand["rerank_score"] = None
            cand["retrieval_score"] = rrf_sc

        candidates.sort(key=lambda item: item["evidence_score"], reverse=True)
        return candidates[:top_k]
