import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import HYDE_SYSTEM_MESSAGE, HYDE_HUMAN_TEMPLATE
from knowledge.utils.bge_client_utils import get_bge_m3_client, generate_dense_and_sparse
from knowledge.utils.llm_client_utils import get_llm_client
from knowledge.utils.milvus_client_utils import hybrid_search

load_dotenv()


class HyDESearchNode(BaseNode):
    name = "HyDE_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        #获取前一个节点的item-names和rewritten_query
        item_names=state.get("item_names")
        if not item_names:
            raise StateFieldError(node_name=self.name,field_name="item_names")

        rewritten_query=state.get("rewritten_query")
        if not rewritten_query:
            raise StateFieldError(node_name=self.name,field_name="rewritten_query")


        #根据item-names和rewritten_query构建提示词，调用大模型，生成假设性文档
        hyde_doc=self.call_llm_generate_hyde_doc(item_names,rewritten_query)
        print(f"假设性文档:{hyde_doc}")


        #拼接rewritten_query和假设性文档，组成查询文本
        hyde_query_context=f"{rewritten_query}\n{hyde_doc}"


        #将查询文本向量化
        bge_m3_client=get_bge_m3_client()
        embedding_result=generate_dense_and_sparse(bge_m3_client,[hyde_query_context])
        dense_vector=embedding_result["dense_vector"][0]
        sparse_vector=embedding_result["sparse_vector"][0]

        #设置标量过滤条件
        filter_expr=self.build_filter_expr(item_names)

        # 执行混合查询
        hybrid_search_result=hybrid_search(
            collection_name=os.getenv("CHUNK_COLLECTION_NAME"),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            dense_weight=0.4,
            sparse_weight=0.6,
            filter_expr=filter_expr,
            output_fields=["chunk_id","content","title","item_name"]
        )

        # 返回结果
        # state["hyde_embedding_chunks"]=hybrid_search_result
        # return state
        hyde_chunks=[]
        for hit in (hybrid_search_result or []):
            entity=hit.get("entity",{})
            chunk={
                **entity,
                "_hybrid_distance":hit.get("distance", 0)
            }
            hyde_chunks.append(chunk)
        return {"hyde_embedding_chunks":hyde_chunks}


    def call_llm_generate_hyde_doc(self, item_names, rewritten_query):

        #初始化客户端
        llm_client=get_llm_client()

        #构建提示词
        systemmessage=HYDE_SYSTEM_MESSAGE
        humanmessage=HYDE_HUMAN_TEMPLATE.format(
            item_names="、".join(item_names) if item_names else "相关产品",
            rewritten_query=rewritten_query
        )
        prompt=[
            SystemMessage(content=systemmessage),
            HumanMessage(content=humanmessage)
        ]

        response=llm_client.invoke(prompt)
        return response.content

    def build_filter_expr(self, item_names):
        item_names_filter=",".join(f'"{name}"' for name in item_names)
        return f"item_names in [{item_names_filter}]"


if __name__=="__main__":
    state={
        "item_names":"室内无线网关",
        "rewritten_query":"怎么验证LA2608 与无线控制器是否连通"
    }

    HyDESearchNode=HyDESearchNode()
    result=HyDESearchNode(state)
    print(result)