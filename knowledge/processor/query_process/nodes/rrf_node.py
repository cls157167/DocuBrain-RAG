"""
RRF (Reciprocal Rank Fusion) 融合排序节点

职责：
  将三路检索结果（HyDE 假设性文档嵌入检索 / Web 网络搜索 / KG 知识图谱）
  通过加权 RRF 算法融合去重，输出单一排序列表到 state["rrf_chunks"]。

三路召回各自的定位：
  ┌──────────┬──────────────────────┬─────────────────────────────────┐
  │ 通路      │ state 字段             │ 召回逻辑                         │
  ├──────────┼──────────────────────┼─────────────────────────────────┤
  │ hyde     │ hyde_embedding_chunks │ query → LLM 生成假设文档 →       │
  │          │                      │ Dense+Sparse 混合检索 → 语义匹配  │
  │ web      │ web_search_docs      │ query → MCP 联网搜索 → 外部信息   │
  │ kg       │ kg_chunks            │ query → LLM 提取实体 → Neo4j      │
  │          │                      │ 图遍历 → 结构化关联匹配           │
  └──────────┴──────────────────────┴─────────────────────────────────┘

RRF 原理：
  RRF 是一种经典的零-shot 排序融合算法，核心思想是"只信排名，不信分数"。
  每条文档的最终得分 = Σ w_s / (k_s + rank_s(d))
  - w_s:     检索通路 s 的权重
  - k_s:     通路 s 的平滑常数，k 越小则头部排名贡献越大
  - rank_s(d): 文档 d 在第 s 路检索中的排名（从 1 开始）

为什么用 RRF 而不是简单拼接：
  1. 各路检索的相似度分数尺度不可比
     - HyDE 混合检索返回的是 Dense+BM25 加权分数
     - Web 搜索返回的是百炼平台的引擎内部排序
     - KG 图遍历的深度 ≠ 语义相关性
  2. RRF 只关心相对排名，天然免疫分数尺度差异
  3. 计算量近乎为零（纯算术运算），不增加 GPU 开销

融合通路与权重/k值设计：
  ┌──────────┬──────┬──────┬──────────────────────────────────────────┐
  │ 通路      │ 权重  │ k值  │ 说明                                      │
  ├──────────┼──────┼──────┼──────────────────────────────────────────┤
  │ hyde     │ 1.0  │ 60   │ 混合检索结果多，k大→平滑噪声、防头部垄断   │
  │ web      │ 1.0  │ 60   │ 联网搜索结果多，k大→同样平滑处理           │
  │ kg       │ 0.7  │ 45   │ 图遍历结果少而精，k适中→适度放大头部信号    │
  └──────────┴──────┴──────┴──────────────────────────────────────────┘

KG 独立 k 值的设计理由：
  KG 召回与 HyDE/Web 有本质不同——图遍历的结果数量少（通常 2~5 条），
  但每一条都经过实体对齐+图结构验证，精确度高。若用统一的 k=60：
    KG #1 贡献 = 0.7/(60+1) = 0.01148
    KG #3 贡献 = 0.7/(60+3) = 0.01111  ← 与 #1 几乎无差别
  改用 k=45 后：
    KG #1 贡献 = 0.7/(45+1) = 0.01522  ← 贡献提升 32%
    KG #3 贡献 = 0.7/(45+3) = 0.01458  ← 排名差异有所体现
  相比 k=60 时头部结果被"压平"，k=45 在不极端倾斜的前提下让 KG 的
  精确信号在融合排序中合理发挥作用。

与下游 Rerank 节点的关系：
  RRF 做"粗排" — 三路合并、去重、按融合分截断（默认 Top-10）
  Rerank 做"精排" — 用 Cross-Encoder 对粗排结果逐条精细打分

去重策略：
  同一文档可能被多个通路召回（如 KG 中查到的 chunk 恰好也在 HyDE 的 Top-N 里）。
  - HyDE / KG 的 chunk 以 chunk_id 为主键去重
  - Web 搜索结果（无 chunk_id）以 snippet 的 MD5 为主键去重
  - 去重时保留首次出现的版本，后续重复只追加来源标记

参考：
  - Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion outperforms Condorcet
    and individual rank learning methods." SIGIR 2009.
"""

import hashlib
import logging
from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState

logger = logging.getLogger(__name__)


class RRFNode(BaseNode):
    """RRF 融合排序节点。

    将 HyDE 语义检索、Web 联网搜索、KG 知识图谱三路召回结果
    通过加权 Reciprocal Rank Fusion 算法合并去重排序。

    三路召回的特点：
      - hyde: 从内部知识库（Milvus）做 Dense+Sparse 混合检索，覆盖面广、查准率高
      - web:  从阿里云百炼平台联网搜索，补充时效性信息和外部知识
      - kg:   从 Neo4j 图数据库做实体关联遍历，擅长关系推理类问题
      Web 结果的结构（snippet/title/url）与 chunk 结构（chunk_id/content/title）不同，
      本节点在去重和融合时统一处理两种结构。

    Attributes:
        name: 节点名称，用于 LangGraph 注册和日志前缀。

    Usage:
        node = RRFNode()
        state = node(state)
        # state["rrf_chunks"] 已填充融合排序后的 Top-N 文档
    """

    name = "rrf_node"

    # ========================================================================
    # 文档唯一标识 — 兼容 chunk 和 web 两种结构
    # ========================================================================

    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        """提取文档的唯一标识符，用于跨通路去重。

        不同通路返回的文档结构不同：
          - HyDE / KG 返回 chunk 结构：含 chunk_id（Milvus 主键）
          - Web 返回搜索结果结构：含 snippet，无 chunk_id

        优先级：
          1. chunk_id  — Milvus 内部主键，最可靠（HyDE / KG 通路）
          2. url       — 网页 URL，对 web 搜索结果有较好的唯一性
          3. snippet / content 的 MD5 — 最终降级方案

        Args:
            doc: 单条文档，结构取决于来源通路。

        Returns:
            文档唯一标识字符串。
        """
        # 优先用 Milvus 主键
        chunk_id = doc.get("chunk_id")
        if chunk_id is not None:
            return f"chunk_{chunk_id}"

        # web 结果优先用 URL（比 MD5 更语义化、可追溯）
        url = doc.get("url")
        if url:
            return f"web_{url}"

        # 最终降级：对文本内容取 MD5
        text = doc.get("snippet") or doc.get("content") or ""
        return f"hash_{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}"

    # ========================================================================
    # 去重与排名索引构建
    # ========================================================================

    def _collect_and_dedup(
        self,
        sources: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, int]]]:
        """收集所有通路的文档，去重，并记录每条文档在各通路中的排名。

        去重规则：
          - 以 _get_doc_id() 返回值为主键
          - 首次出现的文档版本被保留（content/title 等字段以首次为准）
          - 后续出现的同名文档只追加 _rrf_sources 标记，不覆盖内容
          - 同一通路对同一文档多次命中时，只保留最高排名（最小 rank 值）

        这样做的原因：
          如果同一 chunk 同时被 HyDE 和 KG 召回，说明该 chunk 既有语义相关性
          又有图结构关联，综合信号更强 → RRF 分数会因多通路命中而更高。

        Args:
            sources: {来源名称: [文档列表]}，每个列表已按该通路的相关性降序排列。

        Returns:
            (doc_map, rank_map) 二元组：
            - doc_map:  {doc_id: 文档对象}，含内部字段 _rrf_sources
            - rank_map: {doc_id: {来源名称: 排名(从1开始)}}
        """
        doc_map: Dict[str, Dict[str, Any]] = {}
        rank_map: Dict[str, Dict[str, int]] = {}

        for source_name, docs in sources.items():
            if not docs:
                self.logger.debug(f"来源 '{source_name}' 无结果，跳过")
                continue

            self.logger.debug(f"处理来源 '{source_name}': {len(docs)} 条")

            for rank_0based, doc in enumerate(docs):
                if not isinstance(doc, dict):
                    self.logger.warning(
                        f"来源 '{source_name}' 中存在非 dict 类型的文档: "
                        f"{type(doc).__name__}，跳过"
                    )
                    continue

                doc_id = self._get_doc_id(doc)
                rank_1based = rank_0based + 1  # RRF 排名从 1 开始

                # --- 记录文档内容（首次出现为准）---
                if doc_id not in doc_map:
                    doc_copy = dict(doc)
                    doc_copy["_rrf_sources"] = [source_name]
                    doc_map[doc_id] = doc_copy
                else:
                    # 已存在：只追加来源标记（去重）
                    if source_name not in doc_map[doc_id]["_rrf_sources"]:
                        doc_map[doc_id]["_rrf_sources"].append(source_name)

                # --- 记录排名（同一来源只保留最高排名）---
                if doc_id not in rank_map:
                    rank_map[doc_id] = {}
                if source_name not in rank_map[doc_id]:
                    rank_map[doc_id][source_name] = rank_1based
                else:
                    # 保留更小的 rank 值 = 更高排名
                    if rank_1based < rank_map[doc_id][source_name]:
                        rank_map[doc_id][source_name] = rank_1based

        # 统计跨通路召回（同一文档被多路同时命中 → 信号更强）
        multi_source_count = sum(
            1 for doc in doc_map.values()
            if len(doc.get("_rrf_sources", [])) > 1
        )
        if multi_source_count > 0:
            self.logger.info(
                f"跨通路重复文档: {multi_source_count} 条 "
                f"({multi_source_count / max(len(doc_map), 1) * 100:.1f}%)"
            )

        return doc_map, rank_map

    # ========================================================================
    # RRF 分数计算
    # ========================================================================

    def _compute_rrf_scores(
        self,
        rank_map: Dict[str, Dict[str, int]],
        source_weights: Dict[str, float],
        source_k_values: Dict[str, int],
    ) -> Dict[str, float]:
        """计算每条文档的加权 RRF 分数。

        公式：
          RRF_score(d) = Σ_s w_s / (k_s + rank_s(d))

        其中：
          - w_s  = source_weights[s]，通路 s 的权重
          - k_s  = source_k_values[s]，通路 s 的平滑常数
          - rank_s(d) = 文档 d 在通路 s 中的排名，从 1 开始

        k_s 的差异化设计：
          不同通路的结果数量、精确度差异很大：
          - HyDE / Web: 结果多（10~20条）、噪声多 → k=60 平滑处理
          - KG:         结果少（2~5条）、精确度高 → k=45 适度放大头部信号
          统一 k 值会一刀切地压缩 KG 本就稀少的排名差异，使其头部结果
          在融合中无法体现应有价值。

        Args:
            rank_map:       {doc_id: {来源名称: 排名}}
            source_weights:  {来源名称: 权重系数}
            source_k_values: {来源名称: k 值}

        Returns:
            {doc_id: rrf_score}，分数越高越相关。
        """
        rrf_scores: Dict[str, float] = {}

        for doc_id, source_ranks in rank_map.items():
            score = 0.0
            detail_parts: List[str] = []  # 调试用：记录每路贡献

            for source_name, rank in source_ranks.items():
                weight = source_weights.get(source_name, 1.0)
                k = source_k_values.get(source_name, self.config.rrf_k)
                contribution = weight / (k + rank)
                score += contribution
                detail_parts.append(
                    f"{source_name}(r={rank},w={weight},k={k})={contribution:.6f}"
                )

            rrf_scores[doc_id] = score
            self.logger.debug(
                f"doc_id={doc_id} RRF={score:.6f} = {' + '.join(detail_parts)}"
            )

        return rrf_scores

    # ========================================================================
    # 主流程
    # ========================================================================

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行 RRF 融合排序。

        处理步骤：
          1. 从 state 中收集三路检索结果（HyDE / Web / KG）
          2. 文档去重 + 排名索引构建
          3. 计算加权 RRF 分数
          4. 按分数降序 → 截断 Top-N
          5. 注入元数据（rrf_score / rrf_sources / rrf_ranks）
          6. 写入 state["rrf_chunks"]

        Args:
            state: 查询图状态，需包含至少一路非空检索结果。

        Returns:
            更新后的状态，rrf_chunks 字段已填充。
        """
        # ================================================================
        # Step 1: 收集三路检索结果
        # ================================================================
        self.log_step("step1", "收集三路检索结果（HyDE / Web / KG）")

        # 从 state 中安全获取各路结果，确保不为 None
        sources: Dict[str, List[Dict[str, Any]]] = {
            "hyde": state.get("hyde_embedding_chunks") or [],
            "web":  state.get("web_search_docs") or [],
            "kg":   state.get("kg_chunks") or [],
        }

        # 统计各路文档数量
        hyde_count = len(sources["hyde"])
        web_count = len(sources["web"])
        kg_count = len(sources["kg"])
        total_before_dedup = hyde_count + web_count + kg_count

        self.logger.info(
            f"输入统计: hyde={hyde_count}, web={web_count}, kg={kg_count} → "
            f"总计 {total_before_dedup} 条（去重前）"
        )

        # 快速路径：所有通路均无结果
        if total_before_dedup == 0:
            self.logger.warning(
                "三路检索通路均无结果，RRF 返回空列表。"
                "请检查上游 HyDE / Web / KG 检索节点是否正常执行。"
            )
            state["rrf_chunks"] = []
            return state

        # ================================================================
        # Step 2: 去重并建立排名索引
        # ================================================================
        self.log_step("step2", "文档去重并建立各通路排名索引")

        doc_map, rank_map = self._collect_and_dedup(sources)
        unique_count = len(doc_map)
        dedup_ratio = (
            (total_before_dedup - unique_count) / total_before_dedup * 100
            if total_before_dedup > 0 else 0
        )
        self.logger.info(
            f"去重完成: {total_before_dedup} → {unique_count} 条唯一文档 "
            f"(去重率 {dedup_ratio:.1f}%)"
        )

        # ================================================================
        # Step 3: 计算加权 RRF 分数（每条通路独立的 k 值）
        # ================================================================
        self.log_step(
            "step3",
            f"计算加权 RRF 分数 (k: hyde/web={self.config.rrf_k}, kg={self.config.rrf_kg_k})",
        )

        # 权重设计说明：
        #   hyde=1.0  → 混合检索是内部知识库检索的主通路，权重基准
        #   web=1.0   → 联网搜索补充时效信息，权重与主通路平级
        #   kg=0.7    → 图遍历侧重结构关联，语义信号弱于直接检索，
        #                适当降低权重避免结构噪声污染排序
        source_weights = {
            "hyde": 1.0,
            "web":  1.0,
            "kg":   self.config.rrf_kg_weight,
        }

        # k 值差异化设计：
        #   hyde/web 用 rrf_k=60 → 结果多（10~20条），k 大→平滑噪声、防头部垄断
        #   kg       用 rrf_kg_k=45 → 结果少（2~5条）但精确度高，
        #               k 适中→适度放大 Top-1/Top-2 的贡献，让 KG 的精确信号不被淹没
        source_k_values = {
            "hyde": self.config.rrf_k,
            "web":  self.config.rrf_k,
            "kg":   self.config.rrf_kg_k,
        }
        self.logger.info(
            f"通路参数: weights={source_weights}, k_values={source_k_values}"
        )

        rrf_scores = self._compute_rrf_scores(
            rank_map, source_weights, source_k_values
        )

        # 统计分数分布
        if rrf_scores:
            all_scores = list(rrf_scores.values())
            self.logger.info(
                f"RRF 分数范围: [{min(all_scores):.6f}, {max(all_scores):.6f}], "
                f"均值: {sum(all_scores) / len(all_scores):.6f}"
            )

        # ================================================================
        # Step 4: 按 RRF 分数降序排列
        # ================================================================
        self.log_step("step4", "按 RRF 分数降序排列")

        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda did: rrf_scores[did],
            reverse=True,
        )

        # ================================================================
        # Step 5: 截断 Top-N，注入元数据
        # ================================================================
        max_results = self.config.rrf_max_results
        self.log_step("step5", f"截断 Top-{max_results} 并注入元数据")

        rrf_chunks: List[Dict[str, Any]] = []
        for rank_idx, doc_id in enumerate(sorted_doc_ids[:max_results]):
            doc = doc_map[doc_id]

            # 注入 RRF 元数据字段
            doc["rrf_score"] = round(rrf_scores[doc_id], 6)
            # _rrf_sources 是内部标记字段，提升为最终输出字段
            doc["rrf_sources"] = doc.pop("_rrf_sources", [])
            doc["rrf_ranks"] = rank_map.get(doc_id, {})
            doc["rrf_final_rank"] = rank_idx + 1

            rrf_chunks.append(doc)

        # ================================================================
        # Step 6: 详细日志 — 输出 Top-3 的融合细节，便于调试
        # ================================================================
        if rrf_chunks:
            self.logger.info("── RRF Top-3 融合详情 ──")
            for idx, doc in enumerate(rrf_chunks[:3]):
                # 兼容两种结构：chunk 用 title，web 用 snippet[:50]
                label = (
                    doc.get("title")
                    or (doc.get("snippet") or "")[:50]
                    or "N/A"
                )
                sources_str = "+".join(doc.get("rrf_sources", []))
                ranks_str = ", ".join(
                    f"{src}:#{rank}"
                    for src, rank in doc.get("rrf_ranks", {}).items()
                )
                self.logger.info(
                    f"  #{idx + 1} | score={doc['rrf_score']:.6f} | "
                    f"通路=[{sources_str}] | 各通路排名=({ranks_str}) | "
                    f"label={label}"
                )

        # 统计被截断丢弃的文档数
        dropped = max(0, len(sorted_doc_ids) - max_results)
        if dropped > 0:
            self.logger.info(
                f"截断丢弃 {dropped} 条低分文档（超出 Top-{max_results} 限制）"
            )

        self.logger.info(
            f"RRF 融合完成: {total_before_dedup} 条输入 → "
            f"{unique_count} 条去重 → {len(rrf_chunks)} 条输出"
        )

        # ================================================================
        # Step 7: 写入 state
        # ================================================================
        state["rrf_chunks"] = rrf_chunks
        return state


# ========================================================================
# 独立测试
# ========================================================================
if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("RRF 节点独立测试")
    print("=" * 60)

    rrf_node = RRFNode()

    # ------------------------------------------------------------------
    # 场景 1: 正常三路融合（含跨通路重复文档）
    # ------------------------------------------------------------------
    print("\n【场景 1】正常三路融合（含跨通路重复）")
    print("-" * 40)

    test_state_1 = {
        # HyDE 混合检索 → 内部知识库，有 chunk_id
        "hyde_embedding_chunks": [
            {"chunk_id": 1, "content": "LA2608 电源模块额定电压 48V，最大功率 120W",
             "title": "规格书 - 电源参数"},
            {"chunk_id": 2, "content": "电源故障排查步骤：1.检查输入电压 2.检查保险丝",
             "title": "维修手册 - 故障排查"},
            {"chunk_id": 3, "content": "常见故障代码 E01-E10 说明",
             "title": "FAQ - 故障代码"},
        ],
        # Web 联网搜索 → 外部信息，无 chunk_id，有 snippet/url
        "web_search_docs": [
            {"snippet": "LA2608 电源模块常见故障及排查方法汇总",
             "title": "LA2608 故障排查指南", "url": "https://example.com/la2608"},
            {"snippet": "48V 电源模块散热设计规范",
             "title": "电源散热设计", "url": "https://example.com/thermal"},
        ],
        # KG 图检索 → chunk_id=2 与 HyDE 重复（跨通路共识文档）
        "kg_chunks": [
            {"chunk_id": 2, "content": "电源故障排查步骤：1.检查输入电压 2.检查保险丝",
             "title": "维修手册 - 故障排查"},
            {"chunk_id": 4, "content": "电源模块 --[供电]--> 主板 --[依赖]--> 散热风扇",
             "title": "KG - 电源拓扑"},
        ],
    }

    result_1 = rrf_node.process(test_state_1)
    rrf_chunks_1 = result_1["rrf_chunks"]

    print(f"\n输出 {len(rrf_chunks_1)} 条文档:")
    for doc in rrf_chunks_1:
        label = doc.get("title") or (doc.get("snippet") or "")[:40]
        print(f"  #{doc['rrf_final_rank']} "
              f"score={doc['rrf_score']:.6f} "
              f"sources={doc['rrf_sources']} "
              f"ranks={doc['rrf_ranks']} "
              f"label={label}")

    # 预期：chunk_id=2 同时被 hyde 和 kg 命中 → 得分最高，排第一

    # ------------------------------------------------------------------
    # 场景 2: 全部通路为空
    # ------------------------------------------------------------------
    print("\n【场景 2】全部通路为空（边界情况）")
    print("-" * 40)

    test_state_2 = {
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
        "kg_chunks": [],
    }

    result_2 = rrf_node.process(test_state_2)
    rrf_chunks_2 = result_2["rrf_chunks"]
    print(f"输出: {len(rrf_chunks_2)} 条文档（预期 0）")

    # ------------------------------------------------------------------
    # 场景 3: 只有单通路有结果（降级为透传该通路的排名）
    # ------------------------------------------------------------------
    print("\n【场景 3】单通路输入（只有 HyDE，Web 和 KG 均空）")
    print("-" * 40)

    test_state_3 = {
        "hyde_embedding_chunks": [
            {"chunk_id": 10, "content": "文档 A - 最高排名", "title": "A"},
            {"chunk_id": 20, "content": "文档 B - 次高排名", "title": "B"},
            {"chunk_id": 30, "content": "文档 C - 第三排名", "title": "C"},
        ],
        "web_search_docs": [],
        "kg_chunks": [],
    }

    result_3 = rrf_node.process(test_state_3)
    rrf_chunks_3 = result_3["rrf_chunks"]
    print(f"输出 {len(rrf_chunks_3)} 条文档:")
    for doc in rrf_chunks_3:
        print(f"  #{doc['rrf_final_rank']} "
              f"score={doc['rrf_score']:.6f} "
              f"sources={doc['rrf_sources']} "
              f"title={doc.get('title')}")

    # ------------------------------------------------------------------
    # 场景 4: 跨通路共识验证
    # ------------------------------------------------------------------
    print("\n【场景 4】跨通路共识验证")
    print("-" * 40)
    print("文档 A 在 HyDE 排 #1，在 KG 也排 #1 → 跨通路共识，应排最前")
    print("文档 B 只在 HyDE 排 #2 → 单通路命中，应排在 A 后面")

    test_state_4 = {
        "hyde_embedding_chunks": [
            {"chunk_id": 100, "content": "文档 A - 跨通路共识", "title": "共识文档"},
            {"chunk_id": 200, "content": "文档 B - 仅 HyDE 命中", "title": "单通路文档"},
        ],
        "web_search_docs": [],
        "kg_chunks": [
            {"chunk_id": 100, "content": "文档 A - 跨通路共识", "title": "共识文档"},
        ],
    }

    result_4 = rrf_node.process(test_state_4)
    rrf_chunks_4 = result_4["rrf_chunks"]
    for doc in rrf_chunks_4:
        print(f"  #{doc['rrf_final_rank']} "
              f"score={doc['rrf_score']:.6f} "
              f"sources={doc['rrf_sources']} "
              f"title={doc.get('title')}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)