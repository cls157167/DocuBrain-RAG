import logging
import os
from typing import List

from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.utils.normalize_sparse_l2 import normalize_sparse_l2

load_dotenv()
logger=logging.getLogger(__name__)

def get_bge_m3_client():
    try:
        bge_m3_client=BGEM3EmbeddingFunction(
            model_name=os.getenv("BGE_M3_PATH"),
            devices=os.getenv("BGE_DEVICE"),
            use_fp16=False
        )
        return bge_m3_client
    except Exception as e:
        logger.exception(f"bge_m3客户端初始化失败：{e}")
        raise

def generate_dense_and_sparse(bge_m3_client:BGEM3EmbeddingFunction,document:List[str]):

    #获取向量化的结果
    embedding_results=bge_m3_client.encode_documents(document)

    sparse_result = []
    dense_result = []
    for index in range(len(embedding_results["dense"])):

        # 获取稀疏向量

        #获取指针
        sp=embedding_results["sparse"]
        start=sp.indptr[index]
        end=sp.indptr[index+1]

        #获取tokenid
        token_ids=sp.indices[start:end].tolist()

        #获取权重
        weights=sp.data[start:end].tolist()

        #组合tokenid及权重
        sparse_dict = dict(zip(token_ids, weights))
        sparse_vector = normalize_sparse_l2(sparse_dict)
        sparse_result.append(sparse_vector)

        # 获取稠密向量
        dense_vector=embedding_results["dense"][index].tolist()
        dense_result.append(dense_vector)

    return {
        "dense_vector":dense_result,
        "sparse_vector":sparse_result
    }





