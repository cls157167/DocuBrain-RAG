from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.query_process.nodes.HyDE_search_node import HyDESearchNode
from knowledge.processor.query_process.nodes.answer_out_put_node import AnswerOutPutNode
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.kg_search_node import KGSearchNode
from knowledge.processor.query_process.nodes.reranker_node import RerankerNode
from knowledge.processor.query_process.nodes.rrf_node import RRFNode
from knowledge.processor.query_process.nodes.web_search_node import WebSearchNode
from knowledge.processor.query_process.state import QueryGraphState


def route_after_node_bv_answer(state:QueryGraphState)->bool:
    if state.get("answer"):
        return True
    return False

def create_query_graph():
    """创建查询流程图"""

    #1、创建状态图
    workflow=StateGraph(QueryGraphState)

    #2、添加节点
    nodes={
        "item_name_confirm_node":ItemNameConfirmNode(),
        "multi_search":lambda x:x,
        "HyDE_search_node":HyDESearchNode(),
        "web_search_node":WebSearchNode(),
        "kg_search_node":KGSearchNode(),
        "join":lambda x:{},
        "rrf_node":RRFNode(),
        "reranker_node":RerankerNode(),
        "answer_out_put_node":AnswerOutPutNode(),
    }
    for name,node in nodes.items():
        workflow.add_node(name,node)

    #3、添加边
    # 设置入口点
    workflow.set_entry_point("item_name_confirm_node")
    #添加条件边
    workflow.add_conditional_edges(
        "item_name_confirm_node",
        route_after_node_bv_answer,
        {
            False:"multi_search",
            True:"answer_out_put_node"
        }
    )
    #多路搜索分发
    workflow.add_edge("multi_search","HyDE_search_node")
    workflow.add_edge("multi_search","web_search_node")
    workflow.add_edge("multi_search","kg_search_node")
    #多路搜索合并
    workflow.add_edge("HyDE_search_node","join")
    workflow.add_edge("web_search_node","join")
    workflow.add_edge("kg_search_node","join")
    #顺序边
    workflow.add_edge("join","rrf_node")
    workflow.add_edge("rrf_node","reranker_node")
    workflow.add_edge("reranker_node","answer_out_put_node")
    workflow.add_edge("answer_out_put_node",END)

    #4、编译并返回
    return workflow.compile()