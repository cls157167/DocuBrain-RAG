import json
import logging
import os
from logging import exception
from typing import Any

from bson import Int64
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from ray.train.v2.api import result

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.milvus_client_utils import get_milvus_client

load_dotenv()

#主类，向milvus中存储数据
class ChunksSaveToMilvusNode(BaseNode):
    name="chunks_save_to_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks=state.get("chunks")
        if not isinstance(chunks,list) or not chunks:
            raise exception("chunks类型错误或为空",self.name)

        milvus_client=get_milvus_client()
        save_to_milvus=_SaveToMilvus(milvus_client,os.getenv("CHUNK_COLLECTION_NAME"))
        result=save_to_milvus.insert(chunks)
        self.logger.info(f"成功保存{result['insert_count']}条数据到milvus中")
        inserted_ids=result["ids"]

        for chunk,chunk_id in zip(chunks,inserted_ids):
            chunk["chunk_id"]=chunk_id

        self.logger.info(f"成功将{len(inserted_ids)}个ID写入到chunks中")
        # print(json.dumps(state,ensure_ascii=False,indent=2))
        return state



#创建Schema约束
class _CreateSchema:
    @staticmethod
    def create_schema(milvus_client:MilvusClient):
        schema=milvus_client.create_schema()
        schema.add_field(field_name="chunk_id",auto_id=True,is_primary=True,datatype=DataType.INT64)
        schema.add_field(field_name= "content",datatype=DataType.VARCHAR,max_length=65535)
        schema.add_field(field_name= "title",datatype=DataType.VARCHAR,max_length=1000)
        schema.add_field(field_name= "parent_title",datatype=DataType.VARCHAR,max_length=1000)
        schema.add_field(field_name= "chapter_path",datatype=DataType.VARCHAR,max_length=2000)
        schema.add_field(field_name= "file_title",datatype=DataType.VARCHAR,max_length=1000)
        schema.add_field(field_name= "item_name",datatype=DataType.VARCHAR,max_length=1000)
        schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=1024)
        schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="clean_quality_flag",datatype=DataType.VARCHAR,max_length=20)
        schema.add_field(field_name="fuzzy_dedup",datatype=DataType.BOOL)
        schema.add_field(field_name="fuzzy_score",datatype=DataType.FLOAT)

        return schema

#创建索引
class _CreateIndexes:
    @staticmethod
    def create_indexes(milvus_client:MilvusClient):
        index_params=milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="AUTOINDEX",
            metric_type="IP"
        )
        return index_params

#向Collection中保存数据
class _SaveToMilvus:
    MILVUS_SCHEMA_FIELDS = {
        "content", "title", "parent_title", "chapter_path",
        "file_title", "item_name", "dense_vector", "sparse_vector",
        "clean_quality_flag", "fuzzy_dedup", "fuzzy_score"
    }

    def __init__(self,milvus_client:MilvusClient,collection_name:str):
        self.milvus_client=milvus_client
        self.collection_name=collection_name
        self.logger=logging.getLogger(__name__)

    def _filter_chunk(self, chunk: dict) -> dict:
        return {k: v for k, v in chunk.items() if k in self.MILVUS_SCHEMA_FIELDS}

    def insert(self,chunks:list[dict[str,Any]]):
        if not self.milvus_client.has_collection(collection_name=self.collection_name):
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=_CreateSchema.create_schema(self.milvus_client),
                index_params=_CreateIndexes.create_indexes(self.milvus_client)
            )

        filtered = [self._filter_chunk(c) for c in chunks]

        result=self.milvus_client.insert(
            collection_name=self.collection_name,
            data=filtered
        )
        self.logger.info(f"成功插入{len(filtered)}条数据到{self.collection_name}中")
        return result

if __name__ == "__main__":
    from knowledge.processor.import_process.base import setup_logging
    
    setup_logging()
    
    # 测试数据（模拟向量化后的 chunks，包含清洗质量标记和模糊去重字段）
    test_chunks = [
        {
            "content": "挖掘机是一种工程机械，主要用于挖掘土壤和岩石。",
            "title": "概述",
            "parent_title": "",
            "chapter_path": "第一章 产品概述",
            "file_title": "6W100-整本手册",
            "item_name": "挖掘机",
            "dense_vector": [0.1] * 1024,
            "sparse_vector": {1: 0.5, 100: 0.3, 5000: 0.2},
            "clean_quality_flag": "ok",
            "fuzzy_dedup": False,
            "fuzzy_score": 0.0,
        },
        {
            "content": "6W100挖掘机采用液压驱动系统，额定功率120kW，最大转速2200rpm。",
            "title": "技术规格",
            "parent_title": "概述",
            "chapter_path": "第一章 产品概述/1.1 技术规格",
            "file_title": "6W100-整本手册",
            "item_name": "挖掘机",
            "dense_vector": [0.2] * 1024,
            "sparse_vector": {2: 0.6, 200: 0.4},
            "clean_quality_flag": "ok",
            "fuzzy_dedup": False,
            "fuzzy_score": 0.0,
        },
        {
            "content": "设备整机重量约10吨，铲斗容量0.4立方米，最大挖掘深度5.2米。",
            "title": "主要参数",
            "parent_title": "概述",
            "chapter_path": "第一章 产品概述/1.2 主要参数",
            "file_title": "6W100-整本手册",
            "item_name": "挖掘机",
            "dense_vector": [0.3] * 1024,
            "sparse_vector": {3: 0.7, 300: 0.3},
            "clean_quality_flag": "ok",
            "fuzzy_dedup": True,
            "fuzzy_score": 0.87,
        },
        {
            "content": "短文本",
            "title": "备注",
            "parent_title": "附录",
            "chapter_path": "附录A",
            "file_title": "6W100-整本手册",
            "item_name": "挖掘机",
            "dense_vector": [0.4] * 1024,
            "sparse_vector": {4: 0.8, 400: 0.2},
            "clean_quality_flag": "too_short",
            "fuzzy_dedup": False,
            "fuzzy_score": 0.0,
        },
    ]

    # 模拟包含额外元数据字段的 chunk（验证白名单过滤）
    chunk_with_extra_fields = {
        "content": "日常维护包括检查液压油、更换滤芯、清洁散热器等。",
        "title": "维护保养",
        "parent_title": "",
        "chapter_path": "第二章 维护保养",
        "file_title": "6W100-整本手册",
        "item_name": "挖掘机",
        "dense_vector": [0.5] * 1024,
        "sparse_vector": {5: 0.9, 500: 0.1},
        "clean_quality_flag": "ok",
        "fuzzy_dedup": False,
        "fuzzy_score": 0.0,
        "_md5": "abc123def456",
        "_desensitized": True,
        "_fuzzy_duplicate_of": 0,
    }
    test_chunks.append(chunk_with_extra_fields)

    test_state = {
        "chunks": test_chunks
    }

    try:
        node = ChunksSaveToMilvusNode()
        result = node.process(test_state)

        print("\n=== 测试完成 ===")
        print(f"插入数据条数：{len(test_chunks)}")
        print(f"Collection 名称：{os.getenv('CHUNK_COLLECTION_NAME')}")

        for i, chunk in enumerate(result.get("chunks", [])):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"  chunk_id: {chunk.get('chunk_id')}")
            print(f"  title: {chunk.get('title')}")
            print(f"  clean_quality_flag: {chunk.get('clean_quality_flag')}")
            print(f"  fuzzy_dedup: {chunk.get('fuzzy_dedup')}")
            print(f"  fuzzy_score: {chunk.get('fuzzy_score')}")

    except Exception as e:
        logging.exception("测试失败")

