import logging
import os

from dotenv import load_dotenv
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

load_dotenv()
logger=logging.getLogger(__name__)

def get_milvus_client():
    try:
        milvus_client=MilvusClient(
            uri=os.getenv("MILVUS_URL")
        )
        return milvus_client
    except Exception as e:
        logger.exception(f"milvus_client客户端初始化失败：{e}")
        raise


def hybrid_search(collection_name: str,dense_vector,sparse_vector, top_k: int = 10,
                  dense_weight: float = 0.5, sparse_weight: float = 0.5,
                  filter_expr:str=None,output_fields:list=None):

    try:
        milvus_client = get_milvus_client()

        dense_req =AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=top_k
        )

        sparse_req =AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=top_k
        )

        reranker=WeightedRanker(dense_weight,sparse_weight,norm_score=True)

        search_params={
            "collection_name" : collection_name,
            "reqs" : [dense_req, sparse_req],
            "ranker" : reranker,
            "limit" : top_k
        }

        if filter_expr:
            search_params["filter"]=filter_expr

        if output_fields:
            search_params["output_fields"]=output_fields

        results = milvus_client.hybrid_search(**search_params)

        if results and len(results) > 0:
            for hit in results[0]:
                hit["dense_weight"] = dense_weight
                hit["sparse_weight"] = sparse_weight

        logger.info(f"混合检索完成，collection={collection_name}，返回{len(results[0]) if results else 0}条结果")
        return results[0] if results else []

    except Exception as e:
        logger.exception(f"混合检索失败：{e}")
        raise

