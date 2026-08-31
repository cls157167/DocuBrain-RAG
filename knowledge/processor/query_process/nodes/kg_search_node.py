"""
KG 检索节点（查询端）

职责：
  从用户 query 中提取实体 → Neo4j 图遍历 → 获取关联 chunk，
  写入 state["kg_chunks"] 和 state["kg_triples"]。

检索流程：
  query → LLM 提取实体名称
       → Neo4j CONTAINS 模糊匹配候选实体
       → 种子实体 1-2 跳图遍历 → 三元组 + chunk_ids
       → Milvus 按 chunk_id 批量拉取完整 chunk 内容
       → 写入 state

与 HyDE/Web 的互补关系：
  - HyDE: 语义相似匹配（"这段话意思像不像 query"）
  - KG:   结构化关联匹配（"这个实体跟哪些实体有关"）
  - Web:  外部实时信息

设计：
  - Neo4j 不可用或未构建 KG 时自动跳过，返回空结果
  - 实体对齐分两步：LLM 提取 + Neo4j CONTAINS 匹配
"""

import json
import logging
import math
import os
import re
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import MilvusClient

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.llm_client_utils import get_llm_client
from knowledge.utils.milvus_client_utils import get_milvus_client
from knowledge.utils.bge_client_utils import get_bge_m3_client
from knowledge.utils.neo4j_client_utils import (
    get_neo4j_driver,
    search_entities_by_name,
    traverse_from_entities,
)

load_dotenv()
logger = logging.getLogger(__name__)


class KGSearchNode(BaseNode):
    name = "kg_search_node"

    # 检索参数
    DEFAULT_MAX_ENTITIES: int = 5      # 从 query 中最多提取的实体数
    DEFAULT_MAX_HOPS: int = 2          # 图遍历最大跳数
    DEFAULT_MAX_TRIPLES: int = 50      # 最大三元组数
    DEFAULT_MAX_CHUNKS: int = 10        # 最终返回的最大 chunk 数

    # ========================================================================
    # 意图预判 — 零成本正则筛掉纯概念性问题，避免浪费 KG 检索资源
    # ========================================================================
    # KG 擅长的是关系推理类 query，不擅长定义/概念类 query（HyDE 已经覆盖）
    # 预判逻辑：
    #   命中 CONCEPTUAL → 跳过 KG（"什么是X" → HyDE 就够了）
    #   命中 RELATIONAL → 放心走 KG（"X的Y怎么修" → 需要图遍历）
    #   都不命中     → 走 KG（保守策略，宁可多召回不可漏）
    CONCEPTUAL_PATTERNS = [
        re.compile(r'什么是|是什么|啥是|啥叫|介绍一下|解释一下|定义'),
        re.compile(r'有什么用|有什么作用|功能是什么|作用是|干嘛的'),
        re.compile(r'怎么理解|什么意思|含义'),
        re.compile(r'^什么是|^什么叫|^啥是'),        # 明确以定义开头
    ]
    RELATIONAL_PATTERNS = [
        re.compile(r'怎么修|怎么排查|怎么处理|怎么办|如何解决|如何修复'),
        re.compile(r'包含哪些|有哪些|由什么组成|组成|构成'),
        re.compile(r'导致|引起|造成|原因|为什么.*故障|为什么.*坏'),
        re.compile(r'关联|关系|连接|依赖|影响|副作用'),
        re.compile(r'需要什么|需要哪些|用什么.*修|用什么.*检'),
    ]

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # ---- Step 0: 检查 Neo4j ----
        driver = get_neo4j_driver()
        if driver is None:
            self.logger.info("Neo4j 不可用，跳过 KG 检索")
            state["kg_chunks"] = []
            state["kg_triples"] = []
            # return state
            return {"kg_chunks": [], "kg_triples": []}

        # ---- Step 1: 获取 query ----
        rewritten_query = state.get("rewritten_query", "")
        if not rewritten_query:
            rewritten_query = state.get("original_query", "")
        if not rewritten_query:
            self.logger.info("query 为空，跳过 KG 检索")
            # return state
            return {"kg_chunks": [], "kg_triples": []}

        item_names = state.get("item_names", [])

        self.log_step("step1", f"query: {rewritten_query[:100]}...")

        # ---- Step 1.5: 意图预判，筛掉纯概念性问题 ----
        if self._is_conceptual_query(rewritten_query):
            self.logger.info(f"识别为概念性问题，跳过 KG 检索: {rewritten_query[:60]}")
            state["kg_chunks"] = []
            state["kg_triples"] = []
            # return state
            return {"kg_chunks": [], "kg_triples": []}

        # ---- Step 2: LLM 提取实体 ----
        self.log_step("step2", "LLM 从 query 提取实体名称")
        entity_names = self._extract_entity_names(rewritten_query)
        if not entity_names:
            self.logger.info("未从 query 中提取到实体，跳过 KG 检索")
            state["kg_chunks"] = []
            state["kg_triples"] = []
            # return state
            return {"kg_chunks": [], "kg_triples": []}

        self.logger.info(f"提取到实体: {entity_names}")

        # ---- Step 3: Neo4j 实体对齐 ----
        self.log_step("step3", "Neo4j 实体对齐")
        seed_entities = self._align_entities(entity_names, item_names)
        if not seed_entities:
            self.logger.info("未在 KG 中找到匹配实体")
            state["kg_chunks"] = []
            state["kg_triples"] = []
            # return state
            return {"kg_chunks": [], "kg_triples": []}

        self.logger.info(f"对齐到 {len(seed_entities)} 个种子实体: {seed_entities}")

        # ---- Step 4: 图遍历 ----
        self.log_step("step4", f"从种子实体出发 {self.DEFAULT_MAX_HOPS} 跳图遍历")
        traverse_result = traverse_from_entities(
            seed_entities,
            max_hops=self.DEFAULT_MAX_HOPS,
            max_triples=self.DEFAULT_MAX_TRIPLES,
        )

        triples = traverse_result.get("triples", [])
        chunk_ids = traverse_result.get("chunk_ids", [])

        self.logger.info(
            f"图遍历完成: {len(triples)} 个三元组, "
            f"{len(chunk_ids)} 个关联 chunk"
        )

        # ---- Step 5: 从 Milvus 拉取 chunk 内容 ----
        # NOTE: 多取一倍候选（max_chunks * 2），给后续语义相关性打分留余量
        #       避免好 chunk 在 Milvus query 阶段就因为 limit 被截掉
        self.log_step("step5", "按 chunk_id 从 Milvus 拉取完整内容（候选池 ×2）")
        candidate_chunks = self._fetch_chunks_by_ids(
            chunk_ids,
            max_chunks=self.DEFAULT_MAX_CHUNKS * 2,
        )
        self.logger.info(f"拉取到 {len(candidate_chunks)} 个候选 chunk")

        # ---- Step 5.5: BGE-M3 语义相关性打分排序 ----
        # 为什么不用 Cross-Encoder：
        #   - BGE-M3 已被 HyDE 等节点加载为单例，零额外开销
        #   - 批量编码 1 query + N chunks 一次调用完成，毫秒级
        #   - 这一步只给 KG 路建立内部排名，不需要精排精度
        #     真正的精排留给后面的 Rerank 节点统一处理三路结果
        self.log_step("step5.5", "BGE-M3 语义相关性打分排序")
        kg_chunks = self._score_and_rank_chunks(
            candidate_chunks,
            rewritten_query,
            top_k=self.DEFAULT_MAX_CHUNKS,
        )
        self.logger.info(
            f"KG 打分排序完成: 候选 {len(candidate_chunks)} → "
            f"返回 {len(kg_chunks)}，"
            f"最高分 {kg_chunks[0].get('kg_score', 0):.4f}" if kg_chunks else "返回 0"
        )

        # ---- Step 6: 写入 state ----
        # state["kg_chunks"] = kg_chunks
        # state["kg_triples"] = triples
        # return state
        return {"kg_chunks":kg_chunks,"kg_triples":triples}

    # ========================================================================
    # 实体提取（LLM）
    # ========================================================================

    def _extract_entity_names(self, query: str) -> List[str]:
        """
        用 LLM 从 query 中提取实体名称。

        使用项目已有的 ENTITY_EXTRACT_SYSTEM_PROMPT。
        """
        try:
            llm = get_llm_client()
            prompt = [
                SystemMessage(content=ENTITY_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=f"用户问题：{query}\n\n请提取实体名称列表。"),
            ]
            response = llm.invoke(prompt)
            return self._parse_entity_response(response.content)
        except Exception as e:
            self.logger.warning(f"LLM 实体提取失败: {e}")
            return []

    def _parse_entity_response(self, content: str) -> List[str]:
        """解析 LLM 返回的实体列表 JSON。"""
        content = content.strip()
        content = re.sub(r'^```[\s\w]*\n?', '', content)
        content = re.sub(r'\n?```$', '', content)
        content = content.strip()

        try:
            data = json.loads(content)
            if isinstance(data, dict) and "entities" in data:
                return data["entities"][:self.DEFAULT_MAX_ENTITIES]
            if isinstance(data, list):
                return data[:self.DEFAULT_MAX_ENTITIES]
            return []
        except json.JSONDecodeError:
            # 尝试提取 JSON 子串
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    if isinstance(data, dict) and "entities" in data:
                        return data["entities"][:self.DEFAULT_MAX_ENTITIES]
                except json.JSONDecodeError:
                    pass
            self.logger.warning(f"实体列表 JSON 解析失败: {content[:200]}")
            return []

    # ========================================================================
    # 实体对齐（LLM 提取 → Neo4j CONTAINS 匹配）
    # ========================================================================

    def _align_entities(
        self,
        entity_names: List[str],
        item_names: List[str],
    ) -> List[str]:
        """
        将 LLM 提取的实体名对齐到 KG 中的实体节点。

        策略：
          1. Neo4j CONTAINS 模糊匹配（快速粗筛）
          2. 精确匹配优先，模糊匹配作为补充
          3. 可选按 item_name 过滤（限缩到特定产品范围）
        """
        aligned = set()

        for name in entity_names:
            # 精确匹配
            candidates = search_entities_by_name(name, limit=3)
            if not candidates:
                # 尝试分词后的子串匹配
                # 例如 "LA2608无线网关" → 搜索 "LA2608"
                for term in self._tokenize(name):
                    candidates = search_entities_by_name(term, limit=2)
                    if candidates:
                        break

            for c in candidates:
                aligned.add(c.get("name", ""))

        # 如果对齐结果太多，按优先级裁剪
        aligned_list = list(aligned)
        if len(aligned_list) > self.DEFAULT_MAX_ENTITIES * 2:
            aligned_list = aligned_list[:self.DEFAULT_MAX_ENTITIES * 2]

        return aligned_list

    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词：按常见分隔符拆分，返回长度 ≥ 2 的片段。
        用于从复合实体名中提取可搜索的关键词。
        """
        tokens = re.split(r'[\s\-_/,，、。！？]', text)
        result = []
        for t in tokens:
            t = t.strip()
            if len(t) >= 2:
                result.append(t)
            # 混合文本如 "LA2608室内无线网关" → 拆出 "LA2608"
            parts = re.findall(r'[A-Za-z0-9]+', t)
            result.extend(p for p in parts if len(p) >= 2)
        return result

    # ========================================================================
    # 意图预判
    # ========================================================================

    def _is_conceptual_query(self, query: str) -> bool:
        """
        判断 query 是否为纯概念性问题，决定是否跳过 KG 检索。

        判断逻辑（零成本正则，不调 LLM）：
          1. 先检查是否命中 RELATIONAL 模式 → 明确需要 KG，返回 False
          2. 再检查是否命中 CONCEPTUAL 模式 → 明确不需要 KG，返回 True
          3. 都不命中 → 保守策略，返回 False（走 KG，宁可多召回）

        典型场景：
          "什么是LA2608"          → CONCEPTUAL → 跳过KG ✅
          "LA2608电源故障怎么修"   → RELATIONAL → 走KG    ✅
          "LA2608的参数"          → 都不命中   → 走KG    ✅（保守）
        """
        # 先检查关系推理模式（优先级高于概念模式）
        for pat in self.RELATIONAL_PATTERNS:
            if pat.search(query):
                self.logger.debug(f"命中关系推理模式: {pat.pattern}")
                return False

        # 再检查概念性模式
        for pat in self.CONCEPTUAL_PATTERNS:
            if pat.search(query):
                self.logger.debug(f"命中概念性模式: {pat.pattern}")
                return True

        # 都不命中 → 保守走 KG
        return False

    # ========================================================================
    # Milvus 按 ID 取 chunk
    # ========================================================================

    def _fetch_chunks_by_ids(
        self,
        chunk_ids: List[int],
        max_chunks: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        从 Milvus 按 chunk_id 批量拉取 chunk 内容。

        使用 Milvus query API 做标量过滤。
        """
        if not chunk_ids:
            return []

        try:
            milvus_client = get_milvus_client()
            collection_name = os.getenv("CHUNK_COLLECTION_NAME", "")

            # 构建 filter 表达式: chunk_id in [1, 2, 3]
            ids_str = ", ".join(str(cid) for cid in chunk_ids[:max_chunks * 2])
            filter_expr = f"chunk_id in [{ids_str}]"

            results = milvus_client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=["chunk_id", "content", "title",
                              "chapter_path", "item_name"],
                limit=max_chunks,
            )
            return results or []

        except Exception as e:
            self.logger.exception(f"Milvus 按 ID 查询失败: {e}")
            return []

    # ========================================================================
    # BGE-M3 语义相关性打分排序
    # ========================================================================

    def _score_and_rank_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        用 BGE-M3 对 KG 召回的 chunk 做语义相关性打分，按分数降序排列。

        设计理由：
          - KG 召回的是"图结构上有关系"的 chunk，不保证语义相关性
          - 裸 BFS/图遍历顺序不代表相关性，必须在返回前排序
          - 用 BGE-M3（而非 Cross-Encoder）做一次批量编码，低成本建立排名
          - 得分写入 kg_score 字段，供下游 Rerank/调试使用

        Args:
            chunks: 候选 chunk 列表，每个含 content 字段
            query: 用户改写后的查询文本
            top_k: 最终返回的 chunk 数量

        Returns:
            带 kg_score 字段的 chunk 列表，按分数降序，最多 top_k 条
        """
        if not chunks:
            return []

        # 提取所有 chunk 的 content
        texts = [chunk.get("content", "") for chunk in chunks]
        # 过滤掉空文本的 chunk（同时保持索引对应关系）
        valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) >= 10]

        if not valid_indices:
            self.logger.warning("所有 KG chunk 的 content 均为空或过短，无法打分")
            return chunks[:top_k]

        valid_texts = [texts[i] for i in valid_indices]

        # BGE-M3 批量编码：一次调用处理 query + 所有 chunk 文本
        # 比逐个编码快 N 倍，且向量在同一空间内可直接比较
        try:
            bge_client = get_bge_m3_client()
            all_embeddings = bge_client.encode_documents([query] + valid_texts)

            # 提取 query 向量（第 0 个）和 chunk 向量（第 1..N 个）
            query_vec = all_embeddings["dense"][0]
            chunk_vecs = [all_embeddings["dense"][i + 1] for i in range(len(valid_texts))]
        except Exception as e:
            self.logger.exception(f"BGE-M3 批量编码失败: {e}")
            # 降级：编码失败时返回原始顺序的 top-K
            return chunks[:top_k]

        # 计算 query 与每个 chunk 的余弦相似度
        scores: List[Tuple[int, float]] = []  # [(原始索引, 分数), ...]
        for idx_in_valid, chunk_vec in enumerate(chunk_vecs):
            original_idx = valid_indices[idx_in_valid]
            sim = self._cosine_similarity(query_vec, chunk_vec)
            scores.append((original_idx, sim))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)

        # 取 top-K，将分数注入 chunk
        ranked: List[Dict[str, Any]] = []
        for original_idx, score in scores[:top_k]:
            chunk = dict(chunks[original_idx])  # 浅拷贝，避免污染上游
            chunk["kg_score"] = round(score, 4)
            ranked.append(chunk)

        return ranked

    @staticmethod
    def _cosine_similarity(vec_a: list, vec_b: list) -> float:
        """
        计算两个向量的余弦相似度。

        两个 numpy/BGE 返回的 dense 向量，逐元素运算。

        Returns:
            余弦相似度，范围 [-1, 1]。实际 BGE-M3 对中文文本通常在 0.3~0.95 之间。
        """
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ========================================================================
# 独立测试
# ========================================================================
if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    driver = get_neo4j_driver()
    if not driver:
        print("Neo4j 未连接，请先在 .env 中配置 NEO4J_URI/USERNAME/PASSWORD")
        exit(0)

    state = {
        "rewritten_query": "LA2608电源模块故障怎么排查",
        "original_query": "LA2608电源模块故障怎么排查",
        "item_names": ["室内无线网关"],
    }

    node = KGSearchNode()
    result = node.process(state)

    print("\n=== KG 检索测试结果 ===")
    print(f"三元组 ({len(result.get('kg_triples', []))}):")
    for t in result.get("kg_triples", []):
        print(f"  {t['head']} --[{t['relation']}]--> {t['tail']}")

    print(f"\n关联 Chunks ({len(result.get('kg_chunks', []))}):")
    for c in result.get("kg_chunks", []):
        print(f"  [{c.get('chunk_id')}] {c.get('title')}: "
              f"{c.get('content', '')[:80]}...")
