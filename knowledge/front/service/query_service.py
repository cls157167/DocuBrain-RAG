import logging
import uuid
from typing import List, Dict, Any

from knowledge.front.service.task_service import TaskService
from knowledge.front.utils.task_util import TASK_STATUS_PROCESSING, update_task_status, get_task_result
from knowledge.processor.query_process.main_graph import create_query_graph
from knowledge.utils.mongodb_client_utils import get_recent_message, delete_chat_message
from knowledge.utils.sse_util import get_or_create_sse_queue, push_sse_message, SSEEvent

# 节点名称中文映射
NODE_NAME_CN = {
    "item_name_confirm_node": "商品名称识别与问题改写",
    "multi_search": "多路检索分发",
    "HyDE_search_node": "语义检索（HyDE）",
    "web_search_node": "联网搜索",
    "kg_search_node": "知识图谱检索",
    "join": "多路结果汇聚",
    "rrf_node": "粗排融合（RRF）",
    "reranker_node": "精排排序（Reranker）",
    "answer_out_put_node": "答案生成",
}

# 完整检索路径
FULL_PATH = [
    "item_name_confirm_node",
    "multi_search",
    "HyDE_search_node", "web_search_node", "kg_search_node",
    "join",
    "rrf_node",
    "reranker_node",
    "answer_out_put_node",
]

# 快捷路径（已有答案，跳过检索）
SHORTCUT_PATH = [
    "item_name_confirm_node",
    "answer_out_put_node",
]


class QueryService:
    logger=logging.getLogger(__name__)
    def __init__(self,task_service:TaskService):
        self.task_service=task_service


    #生成session_id和task_id
    def generate_session_id(self)->str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())


    def submit_query(self,task_id:str,is_stream:bool):

        #更新任务状态为任务处理中
        update_task_status(task_id,TASK_STATUS_PROCESSING)

        #如果是流式输出，就创建指定task_id对应的SSE队列，用来传递数据
        if is_stream:
            get_or_create_sse_queue(task_id)

    def run_query_graph(self,task_id,session_id,query,is_stream):
        #构建初始状态
        initial_state={
            "task_id":task_id,
            "session_id":session_id,
            "original_query":query,
            "is_stream":is_stream
        }
        #构建图实例
        query_agent=create_query_graph()

        #执行状态图(非流式)
        # query_agent.invoke(initial_state)

        # 执行状态图(流式)
        done_eng = []
        path_nodes = None  # 等第一个节点完成后再确定走哪条路径
        for event in query_agent.stream(initial_state):
            for name, state in event.items():
                cn_name = NODE_NAME_CN.get(name, name)
                done_eng.append(name)
                self.task_service.mark_node_done(task_id, name)
                self.logger.info(f"{task_id}完成了节点{cn_name}")
                # 第一个节点完成后确定路径
                if path_nodes is None:
                    if name == "item_name_confirm_node" and state.get("answer"):
                        path_nodes = list(SHORTCUT_PATH)
                    else:
                        path_nodes = list(FULL_PATH)
                # 计算进行中的节点：路径中未完成的排最前的几个
                remaining_eng = [n for n in path_nodes if n not in done_eng]
                running_cn = [NODE_NAME_CN.get(n, n) for n in remaining_eng[:3]]
                done_cn = [NODE_NAME_CN.get(n, n) for n in done_eng]
                # 推送进度给前端（DONE 事件触发后队列可能已被清理，加保护）
                if is_stream:
                    try:
                        push_sse_message(task_id, SSEEvent.PROGRESS, {
                            "done_list": done_cn,
                            "running_list": running_cn,
                            "status": "processing"
                        })
                    except ValueError:
                        pass

    def get_answer(self, task_id: str) -> str:
        return get_task_result(task_id, "answer", "")

    def get_history(self, session_id: str, limit: int = 50
                      ) -> List[Dict[str, Any]]:
        records = get_recent_message(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        return delete_chat_message(session_id)
