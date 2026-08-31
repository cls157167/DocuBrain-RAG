import json
import logging
import os
import re
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.base import BaseNode, T, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.exceptions import ItemNameConfirmError
from knowledge.prompts.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.bge_client_utils import get_bge_m3_client, generate_dense_and_sparse
from knowledge.utils.llm_client_utils import get_llm_client
from knowledge.utils.milvus_client_utils import hybrid_search
from knowledge.utils.mongodb_client_utils import get_recent_message, save_chat_message

load_dotenv()

class ItemNameConfirmNode(BaseNode):
    """
    1、获取用户输入原始的问题，根据session_id获取最近十条对话记录
    2、保存用户问题，获取message_id
    3、根据用户问题+历史对话，构建提示词，交给LLM，提取商品名，并重构用户问题
    4、根据提取的商品名和重构的问题，交给milvus向量数据库混合检索，对答案进行评分过滤，确认
    5、确认成功，更新item_names、rewritten_query,有候选，给用户选择，没候选，无法识别
    """

    name = "item_name_confirm_node"


    def __init__(self):
        super().__init__()
        self.item_name_llm=ItemNameLLM()
        self.item_name_vector=ItemNameVector()


    def process(self, state: QueryGraphState) -> QueryGraphState:

        try:
            #1、获取用户的原始问题，最近十条对话记录
            self.log_step("step1", "获取用户原始问题和历史对话")
            original_query=state.get("original_query")
            session_id=state.get("session_id")
            chat_history_list=get_recent_message(session_id)
            self.logger.info(f"最近十条对话记录是{chat_history_list}")

            #2、保存用户问题，获取message_id
            self.log_step("step2", "保存用户问题")
            message_id=save_chat_message(
                session_id,
                "user",
                original_query
            )

            #3、通过原始问题以及历史对话，构建提示词，交给大语言模型，提取商品名，并重构用户问题
            self.log_step("step3", "调用LLM提取商品名并重构问题")
            result=self.item_name_llm.retrieve_and_reconstruction(original_query,chat_history_list)
            item_names_list=result.get("item_names",[])
            rewritten_query=result.get("rewritten_query",original_query)
            self.log_step("step3",
                          f"通过大语言模型提取结果: item_names={item_names_list}, rewritten_query={rewritten_query}")

            #4、将提取的商品名进行向量化，并在之前保存在milvus向量数据库中的文档向量集合进行混合检索，评分对齐
            self.log_step("step4", "将大语言模型提取的商品名向量化，并在milvus向量数据库中混合检索，并对齐评分")
            if item_names_list:
                hy_search_result_list=self.item_name_vector._embedding_and_search(item_names_list)
                self.log_step("step4.1",
                              f"将大语言模型提取的商品名向量化，并在milvus向量数据库中混合检索的结果是{hy_search_result_list}")

                confirm,options=self.item_name_vector.item_name_score_ranking(hy_search_result_list)
                self.log_step("step4.2",
                              f"针对混合检索的结果对齐评分，高于0.7的结果是{confirm}，0.6-0.7的结果是{options}")
            else:
                confirm,options=[],[]
                self.logger.info("将大语言模型提取的商品名向量化，并在milvus向量数据库中混合检索,未能检索出有效结果")

            #5、更新state状态
            state["history"]=chat_history_list
            self.update_state(state,confirm,options,rewritten_query)

            return state

        except Exception as e:
            self.logger.exception(f"商品名称确认节点执行失败：{e}")
            raise ItemNameConfirmError(
                message=f"商品名称确认失败：{e}",
                node_name=self.name,
                cause=e
            )

    def update_state(self, state, confirm, options, rewritten_query):
        if confirm:
            state["item_names"]=confirm
            self.log_step("5.1",f"保存的商品名是{state['item_names']}")
            state["rewritten_query"]=rewritten_query
            self.log_step("5.2", f"保存的重构问题是{state['rewritten_query']}")
        elif options:
            newline="\n"
            state["answer"]=f"您想问的是不是以下几个问题：{newline.join(options)}"
        else:
            state["answer"]="当前问题无法识别，请重新提问"


class ItemNameLLM():

    def __init__(self):
        self.logger = logging.getLogger("query.item_name_llm")

    def retrieve_and_reconstruction(self,original_query,chat_history_list)->Dict:

        try:
            #初始化客户端对象
            self.logger.info("初始化LLM客户端")
            llm_client=get_llm_client(response_format=True)

            #构建提示词
            self.logger.info("构建提示词")
            history_message=""
            for message in chat_history_list:
                role=message.get("role")
                text=message.get("text")
                history_message +=f"{role}:{text}\n"

            humanmessage=ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_message,
                query=original_query
            )
            systemmessage="你是一名专业的商品名称提取智能助手,擅长理解用户意图和提取关键信息"

            prompt=[
                SystemMessage(content=systemmessage),
                HumanMessage(content=humanmessage)
            ]

            #调用大模型返回结果
            self.logger.info("调用LLM")
            response=llm_client.invoke(prompt)
            content=response.content.strip()
            self.logger.info(f"LLM原始返回: {content}")

            #调用方法对大模型返回结果进行清洗及反序列化
            self.logger.info("清洗并反序列化LLM返回内容")
            result=self._clean_content(content)
            self.logger.info(f"清洗后结果: {result}")

            return result

        except Exception as e:
            self.logger.exception(f"LLM商品名称提取失败：{e}")
            raise ItemNameConfirmError(
                message=f"LLM商品名称提取失败：{e}",
                node_name="ItemNameLLM",
                cause=e
            )

    def _clean_content(self, content):

        #对数据进行清洗，比如去除```python、```等
        content = content.strip()
        # 移除开头的markdown代码块标记（如 ```json, ```python, ``` 等）
        content = re.sub(r'^```[\s\w]*\n?', '', content)
        # 移除末尾的 ```
        content = re.sub(r'\n?```$', '', content)
        content = content.strip()

        #2、对清洗好的数据进行反序列为json格式
        cleaned_result:Dict[str,Any]= json.loads(content)

        return cleaned_result




class ItemNameVector():
    def _embedding_and_search(self, item_names_list):
        #对item进行向量化
        bge_m3_client=get_bge_m3_client()
        vector_list=generate_dense_and_sparse(bge_m3_client,item_names_list)
        dense_list=vector_list.get("dense_vector")
        sparse_list=vector_list.get("sparse_vector")

        collection_name=os.getenv("ITEM_COLLECTION_NAME")

        search_result_list=[]
        for dense,sparse,item_name in zip(dense_list,sparse_list,item_names_list):
            #得到每一个混合检索的结果
            search_result= hybrid_search(collection_name, dense, sparse,
                                         dense_weight=0.4,sparse_weight=0.6,output_fields=["item_name"])

            #整理混合检索的结果
            z_search_result={
                "order_query_item_names":item_name,
                "matches":[
                {"milvus_item_name":hit["entity"]["item_name"],"score":hit["distance"]}
                for hit in (search_result if search_result else [])
                ]
            }
            search_result_list.append(z_search_result)

        return search_result_list
        """
        results = [
        # 第1个查询的结果列表（我们只传了1条查询，所以只有1个元素）
        [
        # 每条命中结果（Hit）是一个类似字典的对象
        {"id": 101, "distance": 0.95, "entity": {"content": "挖掘机保养手册...", "item_name": "挖掘机"}},
        {"id": 203, "distance": 0.88, "entity": {"content": "液压系统维护...", "item_name": "挖掘机"}},
        {"id": 55,  "distance": 0.82, "entity": {"content": "日常检查项目...", "item_name": "6W100"}},
        ]
        ]
        """

    def item_name_score_ranking(self, hy_search_result_list):
        confirm=[]
        options=[]

        for hy_search_result in hy_search_result_list:

            #抽取用户询问的商品名
            order_query_item_names=hy_search_result.get("order_query_item_names")

            #对每个混合检索结果的分数降序排序,得到降序之后的结果
            matches=sorted(
                hy_search_result.get("matches"),
                key=lambda s : s.get("score"),
                reverse=True
            )

            if not matches:
                continue

            high=[m for m in matches if m["score"]>=float(os.getenv("HIGH_CONFIDENCE_THRESHOLD"))]

            mid=[m for m in matches if float(os.getenv("MID_CONFIDENCE_THRESHOLD"))<=m["score"]<float(os.getenv("HIGH_CONFIDENCE_THRESHOLD"))]

            if high:
                # 高置信: 优先精确匹配，否则取最高分
                max=next(
                    (h for h in high if h["milvus_item_name"]==order_query_item_names),
                    None
                )

                confirm.append(max["milvus_item_name"] if max else high[0]["milvus_item_name"])
            elif mid:
                options.extend(m["milvus_item_name"] for m in mid[:int(os.getenv("MAX_OPTIONS"))])



        return confirm,options






if __name__=="__main__":
    setup_logging()
    state={
        "original_query":"今天天气怎么样"
    }
    incf=ItemNameConfirmNode()
    result=incf.process(state)
    print(result)