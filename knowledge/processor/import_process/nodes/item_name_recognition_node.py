import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import (
    ValidationError, LLMError, EmbeddingError, MilvusError
)
from knowledge.prompts.upload.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.llm_client_utils import get_llm_client
from knowledge.utils.bge_client_utils import get_bge_m3_client
from knowledge.utils.milvus_client_utils import get_milvus_client
from pymilvus import DataType

from knowledge.utils.normalize_sparse_l2 import normalize_sparse_l2

logger = logging.getLogger(__name__)

load_dotenv()

class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition_node"


    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()

        # 第一步：加载state中的数据，并进行参数校验
        self.log_step("step1", "加载state中的数据，并进行参数校验")
        file_title,chunks=self._validate_input(state)

        # 第二步：取出chunks列表中的前K段内容
        self.log_step("step2", f"取出chunks列表中的前{self.config.item_name_chunk_k}段内容")
        context=self._get_part_chunks(chunks)

        # 第三步：根据取出的内容，生成提示词，调用大模型，生成商品名
        self.log_step("step3", "根据取出的前K段内容，生成提示词，调用大模型，生成商品名")
        item_name=self._recognition_item_name_by_llm(file_title,context)

        # 第四步：对生成的商品名进行嵌入（向量化），含稀疏向量和稠密向量
        self.log_step("step4", "对生成的商品名进行嵌入（向量化），含稀疏向量和稠密向量")
        dense,sparse_result=self._embedding_item_name(item_name)

        # 第五步：创建milvus collection，存储向量
        self.log_step("step5", "创建milvus collection，存储商品名向量")
        self._save_to_milvus(file_title, item_name,dense,sparse_result)

        # 第六步：更新state数据
        self.log_step("step6", "更新state数据（item_name等）")
        self._fill_item_name(item_name,state,chunks)
        return state

    def _validate_input(self, state:ImportGraphState):

        chunks=state.get("chunks")
        file_title=state.get("file_title")

        if not chunks or not isinstance(chunks,list):
            raise ValidationError("chunk为空或者不是列表",self.name)

        if not file_title:
            raise ValidationError("文件标题为空",self.name)

        item_name_chunk_k=self.config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k<=0:
            raise ValidationError("item_name_chunk_k为空或者无效",self.name)

        self.logger.info(f"检测到文件{file_title}。对应的切片长度{len(chunks)}")
        return file_title,chunks

    def _get_part_chunks(self,chunks:list):

        total=0
        result=[]

        for index,chunk in enumerate(chunks[:self.config.item_name_chunk_k]):
            if not isinstance(chunk,dict) or not chunk:
                continue
            content=chunk.get("body")
            spices=f"【切片】{index+1}-{content}"

            result.append(spices)
            total +=len(spices)

            if total>self.config.item_name_chunk_size:
                break

        return "\n\n".join(result)[:self.config.item_name_chunk_size]

    def _recognition_item_name_by_llm(self, file_title, context):
        #初始化客户端
        llm_client=get_llm_client()

        if llm_client is None:
            logger.error(f"llm客户端初始化失败，安全回退到标题名:{file_title}")
            return file_title
        #构建提示词
        user_prompt=ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title,context=context)

        messages=[
            SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        try:
            #调用大模型
            llm_response=llm_client.invoke(messages)

            #获取item_name内容
            item_name=getattr(llm_response,"content","")

            if not item_name or item_name.upper()=="UNKNOWN":
                self.logger.error(f"无法识别有效的商品名，安全回退到{file_title}")
                return file_title

            self.logger.info(f"提取到商品名：{item_name}")
            return item_name
        except Exception as e:
            self.logger.error(f"llm调用失败，商品名安全回退到文件名{file_title}")
            return file_title

    def _embedding_item_name(self,item_name):

        try:
            #获取嵌入模型客户端对象
            logger.info("获取嵌入模型客户端对象")
            bge_m3_client=get_bge_m3_client()

            #获取嵌入结果
            logger.info("获取嵌入结果")
            embedding_result=bge_m3_client.encode_documents([item_name])

            #获取稠密和稀疏向量
            #稠密向量
            logger.info("稠密向量")
            dense=embedding_result["dense"][0].tolist()
            #稀疏向量
            logger.info("稀疏向量")
            sparse=embedding_result["sparse"]
            #指针索引
            start_index=sparse.indptr[0]
            end_index=sparse.indptr[1]
            #权重
            weights=sparse.data[start_index:end_index].tolist()
            #tokenID
            tokenIds=sparse.indices[start_index:end_index].tolist()
            sparse_dict=dict(zip(tokenIds,weights))
            sparse_result=normalize_sparse_l2(sparse_dict)
            return dense,sparse_result
        except Exception as e:
            logger.error(f"嵌入商品名{item_name}失败，原因是{str(e)}")
            raise EmbeddingError(f"嵌入商品名{item_name}失败，原因是{str(e)}",self.name)

    def _save_to_milvus(self, file_title, item_name,dense,sparse_result):

        try:
            #参数校验
            if not dense or not sparse_result:
                logger.warning(f"{item_name}向量生成不完整，跳过入库")
                return

            #获取milvus客户端
            milvus_client=get_milvus_client()

            #对collection表格幂等性操作（不存在就创建新的）
            collection_name=os.getenv("ITEM_COLLECTION_NAME")
            if not milvus_client.has_collection(collection_name=collection_name):
                self._create_collection(milvus_client,collection_name)

            #准备数据
            data={
                "file_title":file_title,
                "item_name":item_name,
                "dense_vector":dense,
                "sparse_vector":sparse_result
            }

            #插入到milvus中
            result=milvus_client.insert(
                collection_name=collection_name,
                data=[data]
            )
            self.logger.info(f"{item_name}向量化后的数据已成功保存到milvus中，id是：{result['ids'][0]}")
        except Exception as e:
            self.logger.exception(f"{item_name}向量化后的数据保存到milvus中失败：{e}")
            raise

    def _create_collection(self, milvus_client, collection_name):

        #创建约束（含主键字段约束，标量字段约束，向量字段约束）
        schema=milvus_client.create_schema()
        #主键约束
        schema.add_field(field_name="id",auto_id=True,is_primary=True,datatype=DataType.INT64)
        #标量约束
        schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=500)
        schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=500)
        #向量约束
        schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=1024)
        schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)

        #创建索引（稠密、稀疏）
        index_params=milvus_client.prepare_index_params()
        #稠密索引
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        #稀疏索引
        index_params.add_index(
            field_name="sparse_vector",
            index_type="AUTOINDEX",
            metric_type="IP"
        )

        #创建collection
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合{collection_name}创建成功，并建立了索引")

    def _fill_item_name(self, item_name, state, chunks):
        state["item_name"]=item_name
        for chunk in chunks:
            chunk["item_name"]=item_name



if __name__ == '__main__':
    setup_logging()
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        node = ItemNameRecognitionNode()
        state = {
            "file_title": "6W100-整本手册",
            "chunks": [
                {"body": "本文档描述了6W100型挖掘机的技术参数和使用说明...", "title": "概述", "parent_title": "", "file_title": "6W100-整本手册"},
                {"body": "6W100挖掘机采用液压驱动系统，额定功率120kW...", "title": "技术规格", "parent_title": "概述", "file_title": "6W100-整本手册"},
                {"body": "设备整机重量约10吨，铲斗容量0.4立方米...", "title": "主要参数", "parent_title": "概述", "file_title": "6W100-整本手册"},
            ],
            "item_name": "",
        }
        result = node.process(state)
        print(f"\n最终item_name：{result['item_name']}")
    except Exception:
        logger.exception("节点执行失败")
