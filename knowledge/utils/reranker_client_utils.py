import logging
import os
from typing import List, Tuple

from dotenv import load_dotenv
from FlagEmbedding import FlagReranker

load_dotenv()
logger = logging.getLogger(__name__)

_reranker_client: FlagReranker = None


def get_reranker_client() -> FlagReranker:
    global _reranker_client
    if _reranker_client is None:
        try:
            fp16 = os.getenv("BGE_RERANKER_FP16", "0") == "1"
            _reranker_client = FlagReranker(
                os.getenv("BGE_RERANKER_LARGE"),
                device=os.getenv("BGE_RERANKER_DEVICE", "cpu"),
                use_fp16=fp16,
            )
            logger.info("reranker 模型加载完成")
        except Exception as e:
            logger.exception(f"reranker 客户端初始化失败：{e}")
            raise
    return _reranker_client


def rerank(query: str, documents: List[str]) -> List[Tuple[str, float]]:
    client = get_reranker_client()
    pairs = [[query, doc] for doc in documents]
    scores = client.compute_score(pairs)
    if isinstance(scores, float):
        scores = [scores]
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs
