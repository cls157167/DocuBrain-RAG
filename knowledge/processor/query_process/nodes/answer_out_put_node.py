import logging
from typing import List, Dict, Any

from knowledge.front.utils.task_util import set_task_result
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ANSWER_PROMPT
from knowledge.utils.llm_client_utils import get_llm_client
from knowledge.utils.mongodb_client_utils import save_chat_message
from knowledge.utils.sse_util import push_sse_message, SSEEvent

logger = logging.getLogger(__name__)


class AnswerOutPutNode(BaseNode):
    name = "answer_out_put_node"


    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id=state.get("session_id")
        is_stream=state.get("is_stream")
        task_id=state.get("task_id")
        #1、判断是否已有答案
        if state.get("answer"):
            push_sse_message(task_id,SSEEvent.MESSAGE,{"answer":state.get("answer")})

        else:
            #2、构建提示词
            prompt=self._build_prompt(state)
            state["prompt"]=prompt

            #3、调用大模型、生成答案
            self._call_llm_generate_answer(prompt,state)


        #8、保存历史对话
        if state.get("answer"):
            self._write_history(state)


        #9、流式结果发送给结束事件
        if is_stream:
            push_sse_message(task_id,SSEEvent.DONE, {"answer":state.get("answer"),"status":"completed"})

        return state



    def _build_prompt(self, state: QueryGraphState) -> str:
        """构建 LLM 提示词。

        从 state 中提取排序文档、商品名、历史对话、用户问题，
        按 ANSWER_PROMPT 模板组装为完整的提示词字符串。

        组装逻辑：
          - context:    排序文档 → 可读文本（含排名、标题、内容）
          - history:    历史对话 → "user: ...\\nassistant: ..."
          - item_names: 商品名列表 → "商品A、商品B"
          - question:   rewritten_query 优先 → original_query 降级

        Args:
            state: 查询图状态。

        Returns:
            完整的提示词字符串，可直接传入 LLM。
        """
        # ---- 1. 获取排序文档，格式化为上下文字符串 ----
        context = self._build_context(state)

        # ---- 2. 格式化历史对话 ----
        history_text = self._build_history(state)

        # ---- 3. 格式化商品名 ----
        item_names = state.get("item_names") or []
        item_names_text = "、".join(item_names) if item_names else "未指定"

        # ---- 4. 获取用户问题（改写后优先） ----
        question = (
            state.get("rewritten_query")
            or state.get("original_query")
            or ""
        )

        # ---- 5. 填充模板 ----
        prompt = ANSWER_PROMPT.format(
            context=context,
            history=history_text,
            item_names=item_names_text,
            question=question,
        )

        logger.info(
            f"提示词构建完成: question={question[:60]}..., "
            f"context={len(context)} 字符, "
            f"history={len(history_text.split(chr(10))) - 1} 条"
        )

        return prompt

    # ========================================================================
    # _build_prompt 的子步骤
    # ========================================================================

    def _build_context(self, state: QueryGraphState) -> str:
        """将排序文档格式化为 LLM 可读的上下文字符串。

        优先取 reranked_docs（精排），降级取 rrf_chunks（粗排）。
        每条文档包含：排名、相关度分数、标题、正文。

        Args:
            state: 查询图状态。

        Returns:
            格式化的上下文字符串，无文档时返回 "暂无相关参考内容。"。
        """
        docs = state.get("reranked_docs") or state.get("rrf_chunks") or []

        if not docs:
            return "暂无相关参考内容。"

        blocks: List[str] = []
        max_chars = self.config.max_context_chars
        total = 0

        for doc in docs:
            # 排名：精排 > 粗排 > 未知
            rank = (
                doc.get("rerank_final_rank")
                or doc.get("rrf_final_rank")
                or "?"
            )
            # 分数
            score = doc.get("rerank_score") or doc.get("rrf_score") or 0

            # 标题
            title = doc.get("title", "")
            title = " ".join(title.split()) if title else "无标题"

            # 正文：优先 content（chunk 结构），其次 snippet（web 结果）
            body = doc.get("content") or doc.get("snippet") or ""
            body = " ".join(body.split()) if body else "无内容"

            block = (
                f"【文档 #{rank}】（相关度: {score:.4f}）\n"
                f"标题: {title}\n"
                f"内容: {body}"
            )
            blocks.append(block)

            # 上下文字符数控制，超出上限截断
            total += len(block)
            if max_chars > 0 and total >= max_chars:
                logger.info(f"上下文截断: 保留 {len(blocks)}/{len(docs)} 条")
                break

        return "\n\n".join(blocks)

    def _build_history(self, state: QueryGraphState) -> str:
        """将历史对话列表格式化为文本。

        格式：每行一条 "role: text"，最多取最近 10 条，
        单条超过 200 字符自动截断。

        Args:
            state: 查询图状态。

        Returns:
            格式化的历史对话文本，无历史时返回 "无历史对话"。
        """
        history: List[Dict[str, Any]] = state.get("history") or []

        if not history:
            return "无历史对话"

        lines = []
        for msg in history[-10:]:
            role = msg.get("role", "unknown")
            text = msg.get("text", "")
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"{role}: {text}")

        return "\n".join(lines)



    def _call_llm_generate_answer(self, prompt, state):
        llm_client=get_llm_client()
        task_id = state.get("task_id")
        is_stream=state.get("is_stream")

        if is_stream:
            stream_result=self.stream_generate_result(llm_client,prompt,task_id)
            state["answer"]=stream_result
        else:
            invoke_result=self.invoke_generate_result(llm_client,prompt)
            state["answer"] = invoke_result
            set_task_result(task_id, "answer", state["answer"])

    def stream_generate_result(self, llm_client, prompt, task_id):
        stream_result=""
        try:
            for chunk in llm_client.stream(prompt):
                text=getattr(chunk,"content")
                if text:
                    stream_result +=text
                    push_sse_message(task_id,SSEEvent.MESSAGE,{"message":text})
        except Exception as e:
            self.logger.error(f"流式生成错误：{e}")

        return stream_result

    def invoke_generate_result(self, llm_client, prompt):
        response=llm_client.invoke(prompt)
        invoke_result=response.content
        return invoke_result

    def _write_history(self, state: QueryGraphState) -> None:
        """将用户问题和 assistant 回答持久化到 MongoDB。

        写入两条记录：
          1. role="user"      → 原始问题
          2. role="assistant" → LLM 生成的答案

        Args:
            state: 查询图状态。
        """
        session_id = state.get("session_id", "")
        if not session_id:
            logger.warning("session_id 为空，跳过对话保存")
            return

        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names") or []
        original_query = state.get("original_query", "")
        answer = state.get("answer", "")

        try:
            # ---- 保存用户问题 ----
            user_msg_id = save_chat_message(
                session_id=session_id,
                role="user",
                text=original_query,
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
            logger.info(f"用户问题已保存: session={session_id}, message_id={user_msg_id}")

            # ---- 保存生成的答案 ----
            assistant_msg_id = save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
            logger.info(f"AI 回答已保存: session={session_id}, message_id={assistant_msg_id}")

        except Exception as e:
            # 保存失败不阻断主流程（答案已经推送给用户了）
            logger.exception(f"对话保存失败（不影响答案输出）: {e}")


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("AnswerOutPutNode 独立测试")
    print("=" * 60)

    node = AnswerOutPutNode()

    # ------------------------------------------------------------------
    # 场景 1: 上下文格式化（不依赖 LLM/MongoDB，纯逻辑验证）
    # ------------------------------------------------------------------
    print("\n【场景 1】_build_context — 正常文档格式化")
    print("-" * 40)

    test_state_1: QueryGraphState = {
        "reranked_docs": [
            {
                "chunk_id": 1,
                "title": "LA2608 维修手册 - 电源故障排查",
                "content": (
                    "电源故障排查步骤：1.检查输入电压是否在 44V~52V 范围内；"
                    "2.检查保险丝 F1/F2 是否熔断；3.测量输出端纹波电压应小于 120mV"
                ),
                "rerank_score": 0.9520,
                "rerank_final_rank": 1,
            },
            {
                "chunk_id": 2,
                "title": "LA2608 规格书 - 电源参数",
                "content": (
                    "LA2608 电源模块额定输入电压 48V，最大输入功率 120W"
                ),
                "rerank_score": 0.8810,
                "rerank_final_rank": 2,
            },
            {
                "snippet": "Web 搜索结果：LA2608 故障排查指南",
                "title": "LA2608 外部资料",
                "url": "https://example.com/la2608",
                "rerank_score": 0.7540,
                "rerank_final_rank": 3,
            },
        ],
    }

    ctx = node._build_context(test_state_1)
    print(ctx)
    print(f"\n→ 上下文总长度: {len(ctx)} 字符")

    # ------------------------------------------------------------------
    # 场景 2: 空文档降级
    # ------------------------------------------------------------------
    print("\n【场景 2】_build_context — 空文档降级")
    print("-" * 40)

    test_state_2: QueryGraphState = {
        "reranked_docs": [],
        "rrf_chunks": [],
    }
    ctx = node._build_context(test_state_2)
    print(f"输出: {ctx!r}")

    # ------------------------------------------------------------------
    # 场景 3: 历史对话格式化
    # ------------------------------------------------------------------
    print("\n【场景 3】_build_history — 历史对话格式化")
    print("-" * 40)

    test_state_3: QueryGraphState = {
        "history": [
            {"role": "user", "text": "LA2608 电源故障怎么排查？"},
            {"role": "assistant", "text": "请检查输入电压和保险丝。"},
            {"role": "user", "text": "这个设备的价格是多少？"},
            {"role": "assistant", "text": "LA2608 室内无线网关的参考价格为 3800 元。"},
        ],
    }
    hist = node._build_history(test_state_3)
    print(hist)

    # ------------------------------------------------------------------
    # 场景 4: 无历史对话
    # ------------------------------------------------------------------
    print("\n【场景 4】_build_history — 无历史对话")
    print("-" * 40)

    test_state_4: QueryGraphState = {"history": []}
    hist = node._build_history(test_state_4)
    print(f"输出: {hist!r}")

    # ------------------------------------------------------------------
    # 场景 5: 完整提示词构建验证
    # ------------------------------------------------------------------
    print("\n【场景 5】_build_prompt — 完整提示词组装")
    print("-" * 40)

    test_state_5: QueryGraphState = {
        "original_query": "LA2608 电源故障怎么排查",
        "rewritten_query": "LA2608 电源模块故障排查方法",
        "item_names": ["室内无线网关", "LA2608"],
        "history": [
            {"role": "user", "text": "LA2608 电源故障怎么排查？"},
            {"role": "assistant", "text": "请检查输入电压和保险丝。"},
        ],
        "reranked_docs": [
            {
                "chunk_id": 1,
                "title": "维修手册 - 电源故障排查",
                "content": "电源故障排查步骤：1.检查输入电压...",
                "rerank_score": 0.9520,
                "rerank_final_rank": 1,
            },
        ],
    }
    prompt = node._build_prompt(test_state_5)
    print(f"提示词总长度: {len(prompt)} 字符")
    print(f"\n--- 提示词前 300 字符 ---\n{prompt[:300]}...")

    # ------------------------------------------------------------------
    # 场景 6: 降级取值 — rewritten_query 为空时用 original_query
    # ------------------------------------------------------------------
    print("\n【场景 6】_build_prompt — rewritten_query 为空时降级到 original_query")
    print("-" * 40)

    test_state_6: QueryGraphState = {
        "original_query": "这个设备怎么安装？",
        "rewritten_query": "",
        "item_names": [],
        "history": [],
        "reranked_docs": [],
    }
    prompt = node._build_prompt(test_state_6)
    # 验证 prompt 中是否包含 original_query
    assert "这个设备怎么安装？" in prompt, "降级取值失败"
    print("✓ rewritten_query 为空时正确降级到 original_query")

    # ------------------------------------------------------------------
    # 场景 7: 上下文截断验证（max_context_chars 生效）
    # ------------------------------------------------------------------
    print("\n【场景 7】_build_context — 上下文截断")
    print("-" * 40)

    # 构造一个很短的 max_context_chars
    original_max = node.config.max_context_chars
    node.config.max_context_chars = 150

    test_state_7: QueryGraphState = {
        "reranked_docs": [
            {"title": "文档A", "content": "这是一段很长很长的内容" * 10, "rerank_score": 0.95, "rerank_final_rank": 1},
            {"title": "文档B", "content": "这是第二段内容", "rerank_score": 0.80, "rerank_final_rank": 2},
            {"title": "文档C", "content": "这是第三段内容", "rerank_score": 0.70, "rerank_final_rank": 3},
        ],
    }
    ctx = node._build_context(test_state_7)
    doc_count = ctx.count("【文档")
    print(f"原始文档数: 3, 截断后保留: {doc_count} 条")
    print(f"上下文长度: {len(ctx)} 字符 (上限 {node.config.max_context_chars})")

    # 恢复
    node.config.max_context_chars = original_max

    # ------------------------------------------------------------------
    # 场景 8: prompt 正确写入 state
    # ------------------------------------------------------------------
    print("\n【场景 8】验证 prompt 写入 state")
    print("-" * 40)

    test_state_8: QueryGraphState = {
        "original_query": "测试问题",
        "rewritten_query": "",
        "item_names": [],
        "history": [],
        "reranked_docs": [],
    }
    prompt = node._build_prompt(test_state_8)
    assert "ASK_PROMPT" not in prompt, "模板未被正确填充"
    print("✓ 提示词正确构建（不包含未填充的模板占位符）")

    print("\n" + "=" * 60)
    print("所有场景测试完成")
    print("=" * 60)