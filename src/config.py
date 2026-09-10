"""Cấu hình runtime dùng chung cho ingestion, retrieval và evaluation."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

DEFAULT_EVIDENCE_GATE_THRESHOLD = 1.72


@dataclass(frozen=True)
class IndexConfig:
    """Cấu hình nạp tài liệu, catalog và truy xuất.

    Thuộc tính:
        data_dir (str): Thư mục chứa dữ liệu tài liệu đầu vào (TXT, MD, PDF, DOCX).
        model_dir (str): Thư mục artifact hiện hành được bộ truy xuất đọc trực tiếp.
        catalog_path (str): Catalog là nguồn sự thật về version, ngày hiệu lực và ACL.
        chunk_words (int): Số lượng từ mục tiêu trong một chunk tài liệu.
        overlap_words (int): Số lượng từ gối đầu (overlap) giữa 2 chunk liền kề.
        strategy (str): Chiến lược chunking ("structure_aware" hoặc "sliding_window").
        embedding_model (str): Tên mô hình SentenceTransformers dùng để tạo vector.
        reranker_model (str): Tên mô hình reranker đa ngôn ngữ đã được đánh giá.
        candidate_pool_k (int): Số ứng viên lấy từ mỗi nhánh Dense/BM25.
        rrf_k (int): Hằng số điều chỉnh Reciprocal Rank Fusion.
        rerank_top_k (int): Số ứng viên giữ lại sau reranking.
        evidence_gate_threshold (float): Ngưỡng logit thô mặc định, cần hiệu chỉnh trên Dev.
    """

    data_dir: str = "data/raw"
    model_dir: str = "artifacts"
    catalog_path: str = "configs/knowledge_catalog.yaml"
    chunk_words: int = 250
    overlap_words: int = 40
    strategy: str = "structure_aware"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    candidate_pool_k: int = 30
    rrf_k: int = 60
    rerank_top_k: int = 20
    context_k: int = 4
    evidence_gate_threshold: float = DEFAULT_EVIDENCE_GATE_THRESHOLD

    def validate(self) -> None:
        """Xác thực tính hợp lệ của thông số cấu hình.

        Ngoại lệ:
            ValueError: Nếu các tham số vi phạm ràng buộc kỹ thuật.
        """
        if self.chunk_words <= 0:
            raise ValueError("Kích thước chunk (chunk_words) phải lớn hơn 0.")
        if not (0 <= self.overlap_words < self.chunk_words):
            raise ValueError(
                f"Độ chồng lấp (overlap_words={self.overlap_words}) phải thuộc khoảng [0, chunk_words={self.chunk_words})."
            )
        if self.strategy not in {"structure_aware", "sliding_window"}:
            raise ValueError(
                f"Chiến lược chunking không hợp lệ: '{self.strategy}'. Chọn 'structure_aware' hoặc 'sliding_window'."
            )
        if self.candidate_pool_k <= 0:
            raise ValueError("candidate_pool_k phải lớn hơn 0.")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k phải lớn hơn 0.")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k phải lớn hơn 0.")
        if self.context_k <= 0:
            raise ValueError("context_k phải lớn hơn 0.")
        if not math.isfinite(self.evidence_gate_threshold):
            raise ValueError("evidence_gate_threshold phải là số hữu hạn.")


def parse_args() -> IndexConfig:
    """Trích xuất tham số từ dòng lệnh (CLI) để khởi tạo IndexConfig.

    Kết quả trả về:
        IndexConfig: Đối tượng cấu hình đã qua xác thực.
    """
    parser = argparse.ArgumentParser(
        description="Xây dựng artifact Dense + BM25 cho Vietnamese Policy RAG"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Đường dẫn thư mục chứa tài liệu gốc (mặc định: data/raw)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="artifacts",
        help="Thư mục artifact hiện hành (mặc định: artifacts)",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default="configs/knowledge_catalog.yaml",
        help="Catalog nguồn sự thật về tài liệu, version và ACL",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=250,
        help="Kích thước mục tiêu của mỗi chunk theo số từ (mặc định: 250)",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=40,
        help="Số từ gối đầu giữa các chunk liên tiếp (mặc định: 40)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["structure_aware", "sliding_window"],
        default="structure_aware",
        help="Chiến lược phân tách đoạn văn (mặc định: structure_aware)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Tên hoặc đường dẫn mô hình embedding",
    )
    parser.add_argument(
        "--candidate-pool-k",
        type=int,
        default=30,
        help="Kích thước candidate pool trích xuất từ mỗi nhánh (mặc định: 30)",
    )
    args = parser.parse_args()
    config = IndexConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        catalog_path=args.catalog_path,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        strategy=args.strategy,
        embedding_model=args.embedding_model,
        candidate_pool_k=args.candidate_pool_k,
    )
    config.validate()
    return config
