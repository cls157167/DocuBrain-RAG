import chunk
import json

from dns import name

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import DocumentSplitError, EmbeddingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bge_client_utils import get_bge_m3_client
from knowledge.utils.normalize_sparse_l2 import normalize_sparse_l2_batch, normalize_sparse_l2


class ChunksEmbeddingNode(BaseNode):
    name="chunks_embedding_and_save_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        #从state中加载chunks数据并校验
        chunks=state.get("chunks",[])
        if not isinstance(chunks,list) or not chunks:
            raise DocumentSplitError("文本加载失败，chunks切分异常，请检查",self.name)
        self.log_step("step1",f"开始对{len(chunks)}条切片进行向量化")

        #批量加载数据，并调用嵌入模型对chunks向量化
        #创建嵌入模型客户端
        try:
            m3_client=get_bge_m3_client()
        except Exception as e:
            raise EmbeddingError(f"bge_m3模型初始化失败：{e}",self.name)
        #创建列表，收集增加稠密向量和稀疏向量后的数据
        out_put=[]

        #获取小批量切片，调用嵌入模型进行向量化
        for i in range(0,len(chunks),3):
            batch_chunks=chunks[i:i+3]
            batch_out_put=self.batch_chunks_embedding(batch_chunks,m3_client,i)
            out_put.extend(batch_out_put)

        self.log_step("step2", f"已对{len(chunks)}条切片完成向量化")
        state["chunks"]=out_put
        # print(json.dumps(state, ensure_ascii=False, indent=2))
        return state

    def batch_chunks_embedding(self, batch_chunks, m3_client,start_index):
        """处理一个批次的切片"""
        try:
            #构造要进行嵌入的数据
            self.log_step("step2.1", "构造要进行嵌入的数据:texts=item_name+body")
            texts=[(doc.get("item_name","")+"\n"+
                    doc.get("chapter_path","")+"\n"+
                    doc.get("parent_title","")+"\n"+
                    doc.get("title","")+"\n"+
                    doc.get("body",""))
                   for doc in batch_chunks]

            self.log_step("step2.2", "对texts进行批量嵌入")
            embedding_results=m3_client.encode_documents(texts)
            if not embedding_results:
                self.logger.warning(f"批次{start_index+1}-->{start_index+len(batch_chunks)}未能生成向量")
                return batch_chunks

            out=[]

            self.log_step("step2.3", "获取小批量嵌入后的稠密向量和稀疏向量")
            for j,chunk in enumerate(batch_chunks):
                #稠密向量
                dense_vector=embedding_results["dense"][j].tolist()

                #稀疏向量
                sp=embedding_results["sparse"]
                start=sp.indptr[j]
                end=sp.indptr[j+1]

                token_ids=sp.indices[start:end].tolist()
                weights=sp.data[start:end].tolist()
                sparse_dict=dict(zip(token_ids,weights))
                sparse_vector=normalize_sparse_l2(sparse_dict)


                #组装新的chunk并输出，保留上游节点写入的元数据标记
                new_chunk={
                    "content":chunk.get("body"),
                    "title": chunk.get("title"),
                    "parent_title": chunk.get("parent_title", ""),
                    "chapter_path": chunk.get("chapter_path", ""),
                    "file_title": chunk.get("file_title"),
                    "item_name": chunk.get("item_name"),
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                    "clean_quality_flag": chunk.get("clean_quality_flag", "ok"),
                    "fuzzy_dedup": chunk.get("_fuzzy_dedup", False),
                    "fuzzy_score": chunk.get("_fuzzy_score", 0.0),
                }
                out.append(new_chunk)
            self.logger.info(f"批次{start_index + 1}-->{start_index + len(batch_chunks)}成功生成向量")
            return out
        except Exception as e:
            self.logger.exception(f"批次{start_index}---->{start_index+len(batch_chunks)}处理失败:{e}")
            raise EmbeddingError(f"批次{start_index}---->{start_index+len(batch_chunks)}处理失败，未返回结果",self.name)

if __name__ == "__main__":
    from knowledge.processor.import_process.base import setup_logging
    import logging
    
    setup_logging()
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # 测试数据
    test_state = {
        "chunks": [
            {
                "body": "挖掘机是一种工程机械，主要用于挖掘土壤和岩石。",
                "title": "概述",
                "parent_title": "",
                "file_title": "6W100-整本手册",
                "item_name": "挖掘机"
            },
            {
                "body": "6W100挖掘机采用液压驱动系统，额定功率120kW。",
                "title": "技术规格",
                "parent_title": "概述",
                "file_title": "6W100-整本手册",
                "item_name": "挖掘机"
            },
            {
                "body": "设备整机重量约10吨，铲斗容量0.4立方米。",
                "title": "主要参数",
                "parent_title": "概述",
                "file_title": "6W100-整本手册",
                "item_name": "挖掘机"
            },
            {
                "body": "日常维护包括检查液压油、更换滤芯等。",
                "title": "维护保养",
                "parent_title": "",
                "file_title": "6W100-整本手册",
                "item_name": "挖掘机"
            }
        ]
    }
    
    try:
        node = ChunksEmbeddingNode()
        result = node.process(test_state)
        
        print(f"\n=== 测试完成 ===")
        print(f"原始 chunks 数量：4")
        print(f"输出 chunks 数量：{len(result['chunks'])}")
        
        for i, chunk in enumerate(result['chunks']):
            print(f"\n--- Chunk {i+1} ---")
            print(f"标题：{chunk.get('title')}")
            print(f"稠密向量维度：{len(chunk.get('dense_vector', []))}")
            print(f"稀疏向量非零元素：{len(chunk.get('sparse_vector', {}))}")
            
    except Exception as e:
        logging.exception("测试失败")













