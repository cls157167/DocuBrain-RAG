"""
Reranker 精排节点

职责：
  对 RRF 粗排后的 Top-N 文档用 Cross-Encoder（BGE-Reranker-v2-m3）逐条精细打分，
  通过 gap-based 自适应截断输出最终排序列表到 state["reranked_docs"]。

与上游 RRF 节点的关系：
  RRF 做"粗排" — 三路融合去重，只用排名信息做算术融合，速度快但精度有限
  Rerank 做"精排" — Cross-Encoder 逐对 [query, doc] 深度语义交互打分

为什么 RRF 之后还需要 Rerank：
  1. RRF 只看排名不看内容 — 排名第一的文档语义上可能不如排名第三的
  2. Cross-Encoder 同时编码 query 和 doc，做 token 级交叉注意力，比双塔向量的
     余弦相似度精确得多（但计算量也大得多，所以只用于 Top-N 精排）
  3. RRF 输出通常 10~20 条，Cross-Encoder 推理在 CPU 上也只需几十毫秒

Cross-Encoder vs 双塔模型：
  ┌─────────────────┬──────────────────────┬─────────────────────────┐
  │                  │ Cross-Encoder (精排)  │ 双塔/向量检索 (粗排)     │
  ├─────────────────┼──────────────────────┼─────────────────────────┤
  │ 典型模型          │ bge-reranker-v2-m3   │ bge-m3                  │
  │ 输入              │ [query, doc] 同时输入  │ query 和 doc 分开编码    │
  │ 交互方式          │ Token 级交叉注意力     │ 向量余弦（无交互）       │
  │ 精度              │ 高                    │ 中                      │
  │ 速度              │ 慢（每对都需完整推理）  │ 快（可预编码 + ANN）     │
  │ 适用阶段          │ 精排 Top-10~20         │ 粗排/召回 Top-100~1000  │
  └─────────────────┴──────────────────────┴─────────────────────────┘

Gap-based 自适应截断：
  固定 Top-K 的缺陷：
    如果第 5 名 0.88 分，第 6 名 0.87 分 → 硬截断浪费了相关性接近的好文档
    如果第 3 名 0.85 分，第 4 名 0.30 分 → 后面明显不相关却硬塞到 K 条

  Gap-based 截断规则（按优先级，任一触发即在此处截断）：
    1. 绝对 gap: score[i] - score[i+1] > rerank_gap_abs（默认 0.5）
       → "分数断崖"，后面明显不相关
    2. 相对 gap: (score[i] - score[i+1]) / score[i] > rerank_gap_ratio（默认 0.25）
       → "分数骤降"，后面相关性显著下降
    3. 达到 rerank_max_top_k（默认 10）→ 硬上限
    4. 至少保留 rerank_min_top_k 条（默认 3）→ 软下限，即使有 gap 也不截到 3 以下

  举例：
    分数序列: [0.92, 0.88, 0.85, 0.31, 0.28, 0.25]
    #3→#4 gap: 0.85-0.31=0.54 > 0.5(绝对) 且 0.54/0.85=64% > 25%(相对)
    → 在 #3 处截断，输出 3 条（满足 min_top_k=3）
"""

import logging
from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.exceptions import RerankError
from knowledge.utils.reranker_client_utils import get_reranker_client

logger = logging.getLogger(__name__)


class RerankerNode(BaseNode):
    """Reranker 精排节点。

    用 BGE-Reranker-v2-m3 (FlagEmbedding Cross-Encoder) 对 RRF 粗排结果
    逐条做 [query, doc] 联合编码打分，通过 gap-based 自适应截断确定最终
    返回数量和排序。

    Cross-Encoder 的交叉注意力机制使得模型能看到 query 和 doc 的完整交互，
    对"文档是否真正回答了 query"的判断远比双塔余弦相似度精确。

    Attributes:
        name: 节点名称，用于 LangGraph 注册和日志前缀。
    """

    name = "rerank_node"

    # ========================================================================
    # 文档文本提取 — 兼容 chunk 和 web 两种结构
    # ========================================================================

    def _get_doc_text(self, doc: Dict[str, Any]) -> str:
        """从文档中提取用于 Cross-Encoder 推理的文本。

        不同来源的文档结构不同：
          - HyDE/KG chunk → content + title
          - Web 搜索结果 → snippet + title

        将 title 拼接到 body 前面，给 Cross-Encoder 提供更多上下文信号。
        例如 "维修手册 - 故障排查: 电源故障排查步骤：1.检查..." 比单独的
        content 更能让模型理解文档的主题领域。

        Args:
            doc: 文档 dict，结构取决于来源通路。

        Returns:
            用于 Cross-Encoder 输入的文本。如果 body 和 title 均为空则返回 ""。
        """
        title = doc.get("title", "")
        body = doc.get("content") or doc.get("snippet") or ""

        # 清洗：去除首尾空白，合并多余空格
        title = " ".join(title.split()) if title else ""
        body = " ".join(body.split()) if body else ""

        if title and body:
            # title 和 body 都有 → 拼接，用冒号分隔
            return f"{title}: {body}"
        # 只有一个有内容 → 直接返回有内容的那个
        return title or body

    # ========================================================================
    # Gap-based 自适应截断
    # ========================================================================

    def _gap_truncate(
        self,
        scored_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """根据相邻文档的 Cross-Encoder 分数落差自动决定截断点。

        设计直觉：
          Cross-Encoder 输出的是深度语义相关性分数，同一个 query 下的好文档
          分数通常聚集在高分段（如 0.8~0.95）。当分数出现显著"断崖"时，
          说明后面的文档语义相关性已经大幅下降，不应再喂给 LLM。

        截断算法：
          从第 min_top_k 条开始向后扫描，检查每条与下一条的分数差：
            - 如果分数差触发任一 gap 阈值 → 在此处截断
            - 如果扫描到 max_top_k 也没触发 → 保留 max_top_k 条

        这意味着：
          - 所有文档都很相关（分数集中）→ 保留到 max_top_k 条
          - 中间出现断崖 → 精准截断，不浪费 LLM 上下文窗口
          - 无论怎样至少保留 min_top_k 条

        Args:
            scored_docs: 已按 rerank_score 降序排列的文档列表。

        Returns:
            截断后的文档列表，长度 ∈ [min_top_k, max_top_k]。
        """
        if not scored_docs:
            return []

        max_k = self.config.rerank_max_top_k
        min_k = self.config.rerank_min_top_k
        gap_abs = self.config.rerank_gap_abs
        gap_ratio = self.config.rerank_gap_ratio

        # 先截到 max_top_k
        docs = scored_docs[:max_k]
        n = len(docs)

        # 如果总数不超过 min_k，全部保留
        if n <= min_k:
            return docs

        # 从第 min_k 条开始，逐对检查 gap（保证至少保留 min_k 条）
        cut = n  # 默认全部保留
        for i in range(min_k - 1, n - 1):
            current_score = docs[i]["rerank_score"]
            next_score = docs[i + 1]["rerank_score"]
            diff = current_score - next_score

            # 检测 1: 绝对 gap — 分数直接跳水
            if diff > gap_abs:
                cut = i + 1
                self.logger.info(
                    f"Gap 截断 (绝对): #{i + 1} score={current_score:.4f} → "
                    f"#{i + 2} score={next_score:.4f}, "
                    f"落差={diff:.4f} > 阈值={gap_abs}"
                )
                break

            # 检测 2: 相对 gap — 分数骤降超过比例
            relative_gap = diff / current_score if current_score > 0 else 0.0
            if relative_gap > gap_ratio:
                cut = i + 1
                self.logger.info(
                    f"Gap 截断 (相对): #{i + 1} score={current_score:.4f} → "
                    f"#{i + 2} score={next_score:.4f}, "
                    f"降幅={relative_gap:.1%} > 阈值={gap_ratio:.0%}"
                )
                break

        # 如果触发了截断，统计丢弃的文档分数
        if cut < n:
            dropped_scores = [d["rerank_score"] for d in docs[cut:n]]
            self.logger.info(
                f"截断点: #{cut} (score={docs[cut - 1]['rerank_score']:.4f}), "
                f"丢弃 {n - cut} 条 (分数范围: "
                f"[{min(dropped_scores):.4f}, {max(dropped_scores):.4f}])"
            )

        return docs[:cut]

    # ========================================================================
    # 主流程
    # ========================================================================

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行 Cross-Encoder 精排 + gap-based 自适应截断。

        处理步骤：
          1. 从 state 获取 RRF 粗排结果和 query
          2. 提取每条文档的文本（title + content/snippet）
          3. 构建 [query, doc] 对，用 Cross-Encoder 批量推理打分
          4. 注入 rerank_score，降序排列
          5. Gap-based 自适应截断
          6. 注入最终排名元数据，写入 state["reranked_docs"]

        Args:
            state: 查询图状态，需包含 rrf_chunks 和 query。

        Returns:
            更新后的状态，reranked_docs 字段已填充。
        """
        # ================================================================
        # Step 1: 获取 RRF 粗排结果和 query
        # ================================================================
        self.log_step("step1", "获取 RRF 粗排结果和查询文本")

        rrf_chunks: List[Dict[str, Any]] = state.get("rrf_chunks") or []
        if not rrf_chunks:
            self.logger.warning("RRF 粗排结果为空，跳过 Rerank")
            state["reranked_docs"] = []
            return state

        self.logger.info(f"输入: {len(rrf_chunks)} 条 RRF 粗排文档")

        # query 优先用 rewritten_query（已被 ItemNameConfirm 改写），
        # 降级用 original_query
        query: str = (
            state.get("rewritten_query")
            or state.get("original_query")
            or ""
        )
        if not query:
            self.logger.warning("query 为空，跳过 Rerank，透传 RRF 结果")
            state["reranked_docs"] = rrf_chunks
            return state

        self.logger.info(f"query: {query[:100]}...")

        # ================================================================
        # Step 2: 提取文档文本
        # ================================================================
        self.log_step("step2", "提取文档文本（title + content/snippet）")

        doc_texts: List[str] = []
        empty_count = 0
        for i, doc in enumerate(rrf_chunks):
            text = self._get_doc_text(doc)
            if not text:
                self.logger.warning(
                    f"文档 #{i + 1} (rrf_rank={doc.get('rrf_final_rank')}) "
                    f"无有效文本，将给最低分"
                )
                empty_count += 1
            doc_texts.append(text)

        if empty_count > 0:
            self.logger.warning(
                f"{empty_count}/{len(rrf_chunks)} 条文档无有效文本"
            )

        valid_texts = [t for t in doc_texts if t]
        if not valid_texts:
            self.logger.warning("所有文档均无有效文本，跳过 Rerank")
            state["reranked_docs"] = rrf_chunks
            return state

        # ================================================================
        # Step 3: Cross-Encoder 批量推理打分
        # ================================================================
        self.log_step(
            "step3",
            f"Cross-Encoder 推理: {len(valid_texts)} 对 [query, doc]",
        )

        try:
            reranker = get_reranker_client()
            # 构建 [query, doc_text] 对列表
            pairs = [[query, text] for text in valid_texts]
            raw_scores = reranker.compute_score(pairs)

            # compute_score 对单对返回 float，多对返回 list
            if not isinstance(raw_scores, list):
                raw_scores = [raw_scores]

        except Exception as e:
            self.logger.exception(f"Cross-Encoder 推理失败: {e}")
            raise RerankError(
                message=f"Cross-Encoder 推理失败: {e}",
                node_name=self.name,
                cause=e,
            )

        # ================================================================
        # Step 4: 注入分数，降序排列
        # ================================================================
        self.log_step("step4", "注入 rerank_score 并降序排列")

        # 将分数映射回对应文档（跳过空文本的文档）
        score_idx = 0
        for i, doc in enumerate(rrf_chunks):
            if doc_texts[i]:
                doc["rerank_score"] = round(raw_scores[score_idx], 6)
                score_idx += 1
            else:
                # 无文本的文档给最低分，排在最后
                doc["rerank_score"] = 0.0

        # 按 rerank_score 降序 → rrf_score 降序（同分时粗排高的优先）
        scored_docs = sorted(
            rrf_chunks,
            key=lambda d: (
                d.get("rerank_score", 0.0),
                d.get("rrf_score", 0.0),
            ),
            reverse=True,
        )

        # 分数分布统计
        all_scores = [d["rerank_score"] for d in scored_docs]
        self.logger.info(
            f"Rerank 分数: 范围 [{min(all_scores):.4f}, {max(all_scores):.4f}], "
            f"均值 {sum(all_scores) / len(all_scores):.4f}"
        )

        # ================================================================
        # Step 5: Gap-based 自适应截断
        # ================================================================
        self.log_step(
            "step5",
            f"Gap-based 自适应截断 "
            f"(max={self.config.rerank_max_top_k}, "
            f"min={self.config.rerank_min_top_k}, "
            f"gap_abs={self.config.rerank_gap_abs}, "
            f"gap_ratio={self.config.rerank_gap_ratio})",
        )

        reranked_docs = self._gap_truncate(scored_docs)

        # 注入最终排名
        for i, doc in enumerate(reranked_docs):
            doc["rerank_final_rank"] = i + 1

        # ================================================================
        # Step 6: 详细日志 — 输出最终排序结果
        # ================================================================
        if reranked_docs:
            self.logger.info(
                f"── Rerank 最终结果 ({len(reranked_docs)}/{len(scored_docs)} 条) ──"
            )
            for idx, doc in enumerate(reranked_docs[:5]):
                label = (
                    doc.get("title")
                    or (doc.get("snippet") or "")[:60]
                    or "N/A"
                )
                rrf_score = doc.get("rrf_score", "N/A")
                rrf_rank = doc.get("rrf_final_rank", "?")
                self.logger.info(
                    f"  #{idx + 1} | rerank={doc['rerank_score']:.4f} | "
                    f"RRF rank=#{rrf_rank} score={rrf_score} | "
                    f"label={label}"
                )

        dropped = len(scored_docs) - len(reranked_docs)
        self.logger.info(
            f"Rerank 完成: {len(rrf_chunks)} 条输入 → "
            f"{len(scored_docs)} 条打分 → {len(reranked_docs)} 条输出"
            + (f" (截断丢弃 {dropped} 条)" if dropped else "")
        )

        # ================================================================
        # Step 7: 写入 state
        # ================================================================
        state["reranked_docs"] = reranked_docs
        return state


# ========================================================================
# 独立测试
# ========================================================================
if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("Reranker 节点独立测试")
    print("=" * 60)

    reranker_node = RerankerNode()

    # ------------------------------------------------------------------
    # 场景 1: 正常精排（含 RRF 跨通路共识文档）
    # ------------------------------------------------------------------
    print("\n【场景 1】正常精排 — 验证 Cross-Encoder 是否纠正 RRF 排序")
    print("-" * 40)
    print("chunk_2 在 RRF 中排 #1（跨通路共识），但 Cross-Encoder 可能")
    print("认为 chunk_1 的语义更精准，从而改变排序。")

    test_state_1 = {
        "rewritten_query": "LA2608 电源模块故障怎么排查",
        "original_query": "LA2608 电源模块故障怎么排查",
        "rrf_chunks": [
            {
                "chunk_id": 2,
                "title": "维修手册 - 故障排查",
                "content": "电源故障排查步骤：1.检查输入电压 2.检查保险丝 3.测量输出",
                "rrf_score": 0.031350,
                "rrf_sources": ["hyde", "kg"],
                "rrf_ranks": {"hyde": 2, "kg": 1},
                "rrf_final_rank": 1,
            },
            {
                "chunk_id": 1,
                "title": "规格书 - 电源参数",
                "content": "LA2608 电源模块额定电压 48V，最大功率 120W，工作温度 -20~60°C",
                "rrf_score": 0.016393,
                "rrf_sources": ["hyde"],
                "rrf_ranks": {"hyde": 1},
                "rrf_final_rank": 2,
            },
            {
                "snippet": "LA2608 电源模块常见故障及排查方法汇总",
                "title": "LA2608 故障排查指南",
                "url": "https://example.com/la2608-fix",
                "rrf_score": 0.016393,
                "rrf_sources": ["web"],
                "rrf_ranks": {"web": 1},
                "rrf_final_rank": 3,
            },
            {
                "chunk_id": 3,
                "title": "FAQ - 故障代码",
                "content": "常见故障代码 E01-E10 说明，E01=过压 E02=欠压 E03=过温",
                "rrf_score": 0.015873,
                "rrf_sources": ["hyde"],
                "rrf_ranks": {"hyde": 3},
                "rrf_final_rank": 4,
            },
            {
                "snippet": "48V 电源模块散热设计规范与热管理方案",
                "title": "电源散热设计",
                "url": "https://example.com/thermal",
                "rrf_score": 0.016129,
                "rrf_sources": ["web"],
                "rrf_ranks": {"web": 2},
                "rrf_final_rank": 5,
            },
        ],
    }

    result_1 = reranker_node.process(test_state_1)
    reranked_1 = result_1["reranked_docs"]

    print(f"\n最终输出 {len(reranked_1)} 条文档:")
    for doc in reranked_1:
        label = doc.get("title") or (doc.get("snippet") or "")[:50]
        print(f"  #{doc['rerank_final_rank']} "
              f"rerank={doc['rerank_score']:.4f} | "
              f"RRF曾是#{doc.get('rrf_final_rank')} | "
              f"label={label}")

    # 检查 RRF #1 是否被 Cross-Encoder 保留在第一位
    if reranked_1:
        first = reranked_1[0]
        rrf_rank_of_first = first.get("rrf_final_rank", "?")
        print(f"\n→ RRF #1 的文档在 Rerank 后排第 #{first['rerank_final_rank']}")
        if rrf_rank_of_first == 1 and first["rerank_final_rank"] == 1:
            print("  RRF 和 Rerank 达成共识")
        else:
            print(f"  Cross-Encoder 调整了排序（RRF #{rrf_rank_of_first} → Rerank #1）")

    # ------------------------------------------------------------------
    # 场景 2: 空输入
    # ------------------------------------------------------------------
    print("\n【场景 2】空输入（边界情况）")
    print("-" * 40)

    test_state_2 = {
        "rewritten_query": "测试 query",
        "rrf_chunks": [],
    }

    result_2 = reranker_node.process(test_state_2)
    print(f"输出: {len(result_2['reranked_docs'])} 条文档（预期 0）")

    # ------------------------------------------------------------------
    # 场景 3: Gap 截断验证
    # ------------------------------------------------------------------
    print("\n【场景 3】Gap 截断验证")
    print("-" * 40)
    print("模拟 5 条文档，其中 #3→#4 出现明显分数断崖 → 预期只保留前 3 条")

    # 直接测 _gap_truncate，不调 Cross-Encoder（避免依赖模型文件）
    test_docs_3 = [
        {"title": "文档A", "rerank_score": 0.92, "rrf_score": 0.03},
        {"title": "文档B", "rerank_score": 0.88, "rrf_score": 0.02},
        {"title": "文档C", "rerank_score": 0.85, "rrf_score": 0.01},
        {"title": "文档D", "rerank_score": 0.31, "rrf_score": 0.02},  # ← 断崖！
        {"title": "文档E", "rerank_score": 0.28, "rrf_score": 0.01},
    ]

    truncated = reranker_node._gap_truncate(test_docs_3)
    print(f"输入 5 条 → 截断后 {len(truncated)} 条:")
    for doc in truncated:
        print(f"  {doc['title']}: rerank={doc['rerank_score']:.2f}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
