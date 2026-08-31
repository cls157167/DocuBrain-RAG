"""
KG 构建节点（入库端）

职责：
  在文档完成切分和向量化后，对每个 chunk 调用 LLM 抽取实体和关系，
  写入 Neo4j 构建知识图谱。

设计决策：
  - 放在 chunks_save_to_milvus 之后执行，此时 chunk_id 已分配，可关联 KG 实体与 chunk
  - Neo4j 不可用时自动跳过，不影响主流程
  - 按 batch_size 批量处理，控制 LLM 调用频率
  - 使用 MERGE 保证同名实体唯一性
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.upload.kg_extract_prompt import (
    KG_EXTRACT_SYSTEM_PROMPT,
    KG_EXTRACT_HUMAN_TEMPLATE,
)
from knowledge.utils.llm_client_utils import get_llm_client
from knowledge.utils.neo4j_client_utils import (
    get_neo4j_driver,
    ensure_indexes,
    run_cypher_batch,
)

load_dotenv()
logger = logging.getLogger(__name__)


class KGBuildNode(BaseNode):
    name = "kg_build_node"

    # 批处理参数
    DEFAULT_BATCH_SIZE: int = 3   # 每批处理的 chunk 数
    DEFAULT_MAX_CHUNKS: int = 0   # 最大处理 chunk 数，0 表示不限制

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # ---- 前置检查：Neo4j 是否可用 ----
        driver = get_neo4j_driver()
        if driver is None:
            self.logger.info("Neo4j 不可用，跳过 KG 构建")
            return state

        ensure_indexes()

        # ---- 获取 chunks ----
        chunks = state.get("chunks", [])
        if not chunks:
            self.logger.info("chunks 为空，跳过 KG 构建")
            return state

        item_name = state.get("item_name", "")
        file_title = state.get("file_title", "")

        self.log_step("step1", f"开始 KG 构建，共 {len(chunks)} 个 chunk")

        # ---- 限制处理数量 ----
        if self.DEFAULT_MAX_CHUNKS > 0:
            chunks = chunks[:self.DEFAULT_MAX_CHUNKS]

        # ---- 批量抽取 ----
        total_entities = 0
        total_relations = 0

        for i in range(0, len(chunks), self.DEFAULT_BATCH_SIZE):
            batch = chunks[i:i + self.DEFAULT_BATCH_SIZE]
            self.log_step(
                "step2",
                f"处理 batch {i // self.DEFAULT_BATCH_SIZE + 1}/"
                f"{(len(chunks) - 1) // self.DEFAULT_BATCH_SIZE + 1} "
                f"({i + 1}-{min(i + self.DEFAULT_BATCH_SIZE, len(chunks))})"
            )

            for chunk in batch:
                chunk_id = chunk.get("chunk_id")
                if chunk_id is None:
                    continue

                # LLM 抽取
                result = self._extract_entities(chunk, item_name, file_title)
                if result is None:
                    continue

                entities = result.get("entities", [])
                relations = result.get("relations", [])

                # 写入 Neo4j（一次事务批量提交）
                self._write_chunk_batch(entities, relations, chunk_id, item_name)

                total_entities += len(entities)
                total_relations += len(relations)

        self.logger.info(
            f"KG 构建完成: {total_entities} 个实体, "
            f"{total_relations} 条关系, "
            f"处理 {len(chunks)} 个 chunk"
        )

        return state

    # ========================================================================
    # LLM 抽取
    # ========================================================================

    def _extract_entities(
        self, chunk: Dict, item_name: str, file_title: str
    ) -> Optional[Dict]:
        """
        调用 LLM 从单个 chunk 中抽取实体和关系。
        """
        chunk_text = chunk.get("content", "") or chunk.get("body", "")
        if not chunk_text or len(chunk_text.strip()) < 20:
            return None  # 太短的 chunk 跳过，节省 LLM 调用

        try:
            llm = get_llm_client()
            prompt = [
                SystemMessage(content=KG_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=KG_EXTRACT_HUMAN_TEMPLATE.format(
                    file_title=file_title,
                    item_name=item_name or "未知产品",
                    chunk_text=chunk_text,
                )),
            ]
            response = llm.invoke(prompt)
            return self._parse_llm_response(response.content)
        except Exception as e:
            self.logger.warning(f"LLM 抽取失败 (chunk_id={chunk.get('chunk_id')}): {e}")
            return None

    def _parse_llm_response(self, content: str) -> Optional[Dict]:
        """清洗并解析 LLM 返回的 JSON。"""
        content = content.strip()
        # 去除 markdown 代码块标记
        content = re.sub(r'^```[\s\w]*\n?', '', content)
        content = re.sub(r'\n?```$', '', content)
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取 JSON 子串
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            self.logger.warning(f"LLM 返回 JSON 解析失败: {content[:200]}...")
            return None

    # ========================================================================
    # 写入 Neo4j
    # ========================================================================

    def _write_chunk_batch(
        self,
        entities: List[Dict],
        relations: List[Dict],
        chunk_id: int,
        item_name: str,
    ):
        """
        将单个 chunk 的所有实体和关系收集为 Cypher 语句列表，
        一次 run_cypher_batch 事务提交，避免 N+1 问题。

        之前：每个实体/关系单独一次网络往返（50 实体 = 100+ 次请求）
        现在：整个 chunk 的所有写入合并为 1 次事务提交
        """
        queries: List[tuple] = []

        # ---- 实体写入 ----
        for ent in entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "").strip()
            if not name:
                continue

            # MERGE 实体（同名实体合并，更新类型和产品名）
            queries.append((
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.item_name = $item_name
                """,
                {"name": name, "type": etype, "item_name": item_name},
            ))

            # 实体 → chunk 引用（用于检索时反查源文档）
            queries.append((
                """
                MATCH (e:Entity {name: $name})
                SET e.chunk_id = coalesce(e.chunk_id, []) + $chunk_id
                """,
                {"name": name, "chunk_id": chunk_id},
            ))

        # ---- 关系写入 ----
        for rel in relations:
            head = rel.get("head", "").strip()
            tail = rel.get("tail", "").strip()
            rtype = rel.get("relation", "").strip()
            desc = rel.get("description", "")
            if not head or not tail or not rtype:
                continue

            queries.append((
                """
                MATCH (a:Entity {name: $from_name})
                MATCH (b:Entity {name: $to_name})
                MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
                SET r.description = $description,
                    r.chunk_id = $chunk_id
                """,
                {
                    "from_name": head,
                    "to_name": tail,
                    "rel_type": rtype,
                    "description": desc,
                    "chunk_id": chunk_id,
                },
            ))

        # 一次事务提交全部
        if queries:
            success = run_cypher_batch(queries)
            if not success:
                self.logger.warning(
                    f"chunk_id={chunk_id} 批量写入失败 "
                    f"({len(entities)} 实体, {len(relations)} 关系)"
                )

    # ========================================================================
    # 辅助方法
    # ========================================================================


# ========================================================================
# 独立测试
# ========================================================================
if __name__ == "__main__":
    from knowledge.processor.import_process.base import setup_logging
    setup_logging()

    driver = get_neo4j_driver()
    if not driver:
        print("Neo4j 未连接，请先在 .env 中配置 NEO4J_URI/USERNAME/PASSWORD")
        exit(0)

    test_state = {
        "chunks": [
            {
                "chunk_id": 99901,
                "content": "LA2608室内无线网关采用12V DC供电，额定功率5W，工作温度-20℃到60℃。设备包含一个主控模块和一个射频天线模块。",
                "title": "技术参数",
            },
            {
                "chunk_id": 99902,
                "content": "如果设备指示灯红灯常亮，说明电源模块故障。请使用万用表检测电源输出电压是否为12V。若电压异常，更换电源适配器。",
                "title": "故障排查",
            },
        ],
        "item_name": "室内无线网关",
        "file_title": "LA2608-产品手册",
    }

    node = KGBuildNode()
    result = node.process(test_state)

    print("\n=== KG 构建测试完成 ===")

    # 验证
    from knowledge.utils.neo4j_client_utils import search_entities_by_name, traverse_from_entities
    found = search_entities_by_name("LA2608")
    print(f"搜索 'LA2608': {found}")

    if found:
        result = traverse_from_entities([found[0]["name"]])
        print(f"图遍历结果:")
        for t in result["triples"]:
            print(f"  {t['head']} --[{t['relation']}]--> {t['tail']}")

    # 清理
    from knowledge.utils.neo4j_client_utils import clear_graph
    clear_graph("室内无线网关")
    from knowledge.utils.neo4j_client_utils import close_neo4j_driver
    close_neo4j_driver()
