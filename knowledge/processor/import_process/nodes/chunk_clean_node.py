"""
Chunk 清洗节点（第4-5层 + 质量标记）

在文档切分后对 chunk 列表执行：
  第4层 去重层 — 精确去重(MD5) + 模糊去重(默认关闭，只标记不删，rerank降权)
  第5层 脱敏层 — 手机号/邮箱/身份证正则掩码，失败写入隔离表，不静默入库
  质量标记 — 为每个 chunk 打 clean_quality_flag: ok / too_short / empty

插入位置：document_split_node → chunk_clean_node → item_name_recognition_node
"""

import hashlib
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Set

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState

logger = logging.getLogger(__name__)

# ============================================================================
# 第5层：脱敏正则模式（编译一次，全局复用）
# ============================================================================
# 手机号：1[3-9]\d{9} → 保留前3后4
_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)")
# 邮箱：保留首字符和域名
_EMAIL_RE = re.compile(
    r"([a-zA-Z0-9._%+-])[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
)
# 身份证：18位 → 保留前6后4
_ID_CARD_RE = re.compile(r"(?<!\d)(\d{6})\d{8}(\d{4})(?!\d)")


class ChunkCleanNode(BaseNode):
    """
    Chunk 级清洗节点

    处理去重（L4）、脱敏（L5）、质量标记。
    每个步骤通过 config 独立开关控制。
    """

    name = "chunk_clean_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """从 state 获取 chunks 列表，执行 L4-L5 清洗并打质量标记。"""
        chunks: List[Dict[str, Any]] = state.get("chunks", [])
        if not chunks:
            self.logger.info("chunks 为空，跳过 chunk 级清洗")
            return state

        total_in = len(chunks)

        # ---- 第4层：去重 ----
        if getattr(self.config, "clean_layer4_enabled", True):
            chunks = self._layer4_dedup(chunks)

        # ---- 第5层：脱敏 ----
        if getattr(self.config, "clean_layer5_enabled", True):
            chunks = self._layer5_desensitize(chunks)

        # ---- 质量标记 ----
        chunks = self._assign_quality_flags(chunks)

        self.logger.info(
            f"Chunk 清洗完成: 输入 {total_in} → 输出 {len(chunks)} "
            f"(减少 {total_in - len(chunks)})"
        )

        state["chunks"] = chunks
        return state

    # ========================================================================
    # 第4层：去重层
    # ========================================================================

    def _layer4_dedup(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """第4层：去重。精确去重(MD5)直接删除，模糊去重默认只标记不删。"""
        self.log_step("L4", "去重层")

        # 精确去重
        if getattr(self.config, "clean_exact_dedup", True):
            self.log_step("L4.1", "精确去重 (MD5)")
            chunks = self._exact_dedup(chunks)

        # 模糊去重
        if getattr(self.config, "clean_fuzzy_dedup", False):
            self.log_step("L4.2", "模糊去重（标记模式）")
            threshold = getattr(self.config, "clean_fuzzy_dedup_threshold", 0.85)
            action = getattr(self.config, "clean_fuzzy_dedup_action", "mark")
            chunks = self._fuzzy_dedup(chunks, threshold, action)

        return chunks

    def _exact_dedup(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """精确去重：对 chunk['body'] 计算 MD5，完全重复的只保留第一个。"""
        seen: Set[str] = set()
        deduped = []
        removed = 0
        for chunk in chunks:
            body = chunk.get("body", "")
            md5 = hashlib.md5(body.encode("utf-8", errors="replace")).hexdigest()
            if md5 in seen:
                removed += 1
                self.logger.debug(f"精确去重: 删除重复 chunk (title={chunk.get('title', 'N/A')[:40]})")
                continue
            seen.add(md5)
            chunk["_md5"] = md5
            deduped.append(chunk)
        if removed:
            self.logger.info(f"精确去重: 删除 {removed} 个重复 chunk (剩余 {len(deduped)})")
        return deduped

    def _fuzzy_dedup(
        self, chunks: List[Dict[str, Any]], threshold: float, action: str
    ) -> List[Dict[str, Any]]:
        """
        模糊去重：对相似度超过 threshold 的 chunk 做标记处理。

        使用 Jaccard 相似度（字符 2-gram 集合），处理超长文档时做高效近似。
        action='mark'：打 _fuzzy_dedup=True + _fuzzy_score，在 rerank 阶段降权。
        action='delete'：直接删除相似 chunk。
        """
        if len(chunks) <= 1:
            return chunks

        def _char_bigrams(text: str) -> Set[str]:
            return {text[i:i + 2] for i in range(len(text) - 1)}

        def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        # 预计算所有 chunk 的 bigram 集合
        bigram_sets = [_char_bigrams(c.get("body", "")) for c in chunks]

        marked = 0
        seen = set()
        for i in range(len(chunks)):
            if i in seen:
                continue
            key_i = frozenset(bigram_sets[i])
            for j in range(i + 1, len(chunks)):
                if j in seen:
                    continue
                sim = _jaccard(bigram_sets[i], bigram_sets[j])
                if sim >= threshold:
                    chunks[j]["_fuzzy_dedup"] = True
                    chunks[j]["_fuzzy_score"] = round(sim, 4)
                    chunks[j]["_fuzzy_duplicate_of"] = i
                    marked += 1
                    if action == "delete":
                        seen.add(j)

        if action == "delete" and marked:
            chunks = [c for c in chunks if not c.get("_fuzzy_dedup")]
            self.logger.info(f"模糊去重(delete): 删除 {marked} 个相似 chunk")
        elif marked:
            self.logger.info(f"模糊去重(mark): 标记 {marked} 个相似 chunk (留待 rerank 降权)")

        return chunks

    # ========================================================================
    # 第5层：脱敏层
    # ========================================================================

    def _layer5_desensitize(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """第5层：敏感信息脱敏。失败不静默，按配置写入隔离表或告警。"""
        if not getattr(self.config, "clean_desensitize", True):
            return chunks

        self.log_step("L5", "脱敏层")

        fail_action = getattr(self.config, "clean_desensitize_fail_action", "isolate")
        isolation_enabled = getattr(self.config, "clean_desensitize_isolation_enabled", False)

        stats = {"手机号": 0, "邮箱": 0, "身份证": 0}
        failures = 0

        for idx, chunk in enumerate(chunks):
            body = chunk.get("body", "")
            if not body:
                continue

            try:
                masked_body = body

                if getattr(self.config, "clean_desensitize_phone", True):
                    count_before = len(_PHONE_RE.findall(masked_body))
                    masked_body = _PHONE_RE.sub(r"\1****\2", masked_body)
                    stats["手机号"] += count_before

                if getattr(self.config, "clean_desensitize_email", True):
                    count_before = len(_EMAIL_RE.findall(masked_body))
                    masked_body = _EMAIL_RE.sub(r"\1***@\2", masked_body)
                    stats["邮箱"] += count_before

                if getattr(self.config, "clean_desensitize_id_card", True):
                    count_before = len(_ID_CARD_RE.findall(masked_body))
                    masked_body = _ID_CARD_RE.sub(r"\1********\2", masked_body)
                    stats["身份证"] += count_before

                if masked_body != body:
                    chunk["body"] = masked_body
                    chunk["_desensitized"] = True

            except Exception as e:
                failures += 1
                self.logger.error(f"脱敏失败 chunk[{idx}]: {e}")
                if fail_action == "isolate" and isolation_enabled:
                    chunk["_desensitize_failed"] = True
                    chunk["_desensitize_error"] = str(e)
                elif fail_action == "skip":
                    chunk["_skip"] = True
                # "warn": 仅告警，继续使用原始内容

        # ---- L5 汇总日志 ----
        total = sum(stats.values())
        if total:
            detail = " | ".join(f"{k}: {v}条" for k, v in stats.items() if v)
            msg = f"L5 脱敏层: 掩码 {total} 条敏感信息 ({detail})"
            if failures:
                msg += f" | 失败 {failures} 条"
                if fail_action == "isolate" and isolation_enabled:
                    msg += " (已标记隔离)"
            self.logger.info(msg)
        elif failures:
            self.logger.info(f"L5 脱敏层: 无敏感信息命中 | 脱敏失败 {failures} 条")
        else:
            self.logger.info("L5 脱敏层: 无敏感信息命中")

        return chunks

    # ========================================================================
    # 质量标记
    # ========================================================================

    def _assign_quality_flags(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为每个 chunk 打 clean_quality_flag：
        - "ok": 正常 chunk（长度 ≥ clean_min_chunk_length）
        - "too_short": 长度低于阈值
        - "empty": 空内容 chunk

        empty chunk 根据 clean_empty_chunk_action 处理：
        - "drop": 静默丢弃（默认）
        - "raise": 抛异常，阻断入库流程
        - "keep": 保留空字符串，标记但不删除
        """
        min_length = getattr(self.config, "clean_min_chunk_length", 10)
        action = getattr(self.config, "clean_empty_chunk_action", "drop")

        flagged = []
        stats = {"ok": 0, "too_short": 0, "empty": 0}

        for chunk in chunks:
            body = chunk.get("body", "")

            if not body or not body.strip():
                stats["empty"] += 1
                chunk["clean_quality_flag"] = "empty"

                if action == "raise":
                    raise ValueError(
                        f"发现空 chunk: title={chunk.get('title', 'N/A')}, "
                        f"file={chunk.get('file_title', 'N/A')}"
                    )
                elif action == "drop":
                    continue  # 静默丢弃，不进入下游
                else:  # "keep"
                    flagged.append(chunk)

            elif len(body) < min_length:
                stats["too_short"] += 1
                chunk["clean_quality_flag"] = "too_short"
                flagged.append(chunk)

            else:
                stats["ok"] += 1
                chunk["clean_quality_flag"] = "ok"
                flagged.append(chunk)

        self.logger.info(
            f"质量标记: ok={stats['ok']}, too_short={stats['too_short']}, "
            f"empty={stats['empty']}({'丢弃' if action == 'drop' else action})"
        )
        return flagged


# ============================================================================
# 独立测试入口
# ============================================================================

if __name__ == "__main__":
    setup_logging()
    logging.getLogger().setLevel(logging.DEBUG)

    test_chunks = [
        {"body": "这是第一个chunk的内容，包含型号RS-12的说明。", "title": "chunk1",
         "parent_title": "## 概述", "file_title": "测试手册"},
        {"body": "这是第一个chunk的内容，包含型号RS-12的说明。", "title": "chunk2",
         "parent_title": "## 概述", "file_title": "测试手册"},  # ← MD5重复
        {"body": "这是第一个chunk的内容，包含型号RS-12的说明。", "title": "chunk3",
         "parent_title": "## 概述", "file_title": "测试手册"},  # ← MD5重复
        {"body": "联系方式：13812345678 邮箱：test@example.com 身份证：110101199001011234",
         "title": "chunk4", "parent_title": "## 联系信息", "file_title": "测试手册"},
        {"body": "短", "title": "chunk5",
         "parent_title": "## 其他", "file_title": "测试手册"},  # ← too_short
        {"body": "", "title": "chunk6",
         "parent_title": "## 附录", "file_title": "测试手册"},  # ← empty → 丢弃
    ]

    node = ChunkCleanNode()
    result = node.process({"chunks": test_chunks})

    print("=" * 60)
    print("【Chunk 级清洗结果】")
    for c in result.get("chunks", []):
        flag = c.get("clean_quality_flag", "?")
        desen = "✓" if c.get("_desensitized") else "-"
        md5 = c.get("_md5", "")[:8] if c.get("_md5") else "-"
        fuzzy = f"dup_of_{c.get('_fuzzy_duplicate_of')}" if c.get("_fuzzy_dedup") else "-"
        body_preview = c.get("body", "")[:60]
        skip = "SKIP" if c.get("_skip") else ""
        print(f"  [{flag}] {c.get('title')} | md5={md5} | desen={desen} | "
              f"fuzzy={fuzzy} {skip}")
        print(f"         body: {body_preview}...")
    print("=" * 60)
