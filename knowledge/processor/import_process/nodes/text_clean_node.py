"""
文本清洗节点 — 文档级（第1-3层 + 第6层）

对 PDF 转换后的 Markdown 内容进行分层清洗，在切分之前执行。
切分后的 chunk 级清洗（第4-5层 + 质量标记）由 chunk_clean_node 负责。

六层架构分工：
  第1层 编码层 — 统一换行符、删控制字符和零宽字符（基本不会误杀）
  第2层 空白层 — 压缩连续空行、去行首尾空白、合并错误断行（保护结构化内容）
  第3层 结构噪声层 — 去页眉页脚/页码/导航文本（保护安全警告类重复内容）
  第4层 去重层 ─┐
  第5层 脱敏层 ─┤→ chunk_clean_node（切分后执行）
  质量标记    ─┘
  第6层 术语归一化 — 繁简统一、全角半角统一、型号别名映射

插入位置：md_Img_node → text_clean_node → document_split_node → chunk_clean_node → ...
"""

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState

logger = logging.getLogger(__name__)


# ============================================================================
# 第1层辅助：零宽字符集
# ============================================================================
ZERO_WIDTH_CHARS = {
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
    "­",  # SOFT HYPHEN
    "‎",  # LEFT-TO-RIGHT MARK
    "‏",  # RIGHT-TO-LEFT MARK
    "⁠",  # WORD JOINER
    "⁡",  # FUNCTION APPLICATION
    "⁢",  # INVISIBLE TIMES
    "⁣",  # INVISIBLE SEPARATOR
    "⁤",  # INVISIBLE PLUS
    "᠎",  # MONGOLIAN VOWEL SEPARATOR
    " ",  # NO-BREAK SPACE（替换为普通空格）
}

# ============================================================================
# 第3层辅助：安全关键词默认集
# ============================================================================
DEFAULT_SAFETY_KEYWORDS = {"安全", "警告", "注意", "危险", "严禁", "必须", "重要", "须知", "警示", "紧急"}
DEFAULT_NOISE_KEYWORDS = {"页码", "页眉", "页脚", "版权所有", "版权", "http", "www", "第", "共", "转第"}

# ============================================================================
# 第6层辅助：繁体→简体轻量映射（opencc 不可用时的降级方案）
# ============================================================================
_TRAD_TO_SIMP_MAP: Dict[str, str] = {
    "個": "个", "們": "们", "萬": "万", "與": "与", "專": "专",
    "業": "业", "東": "东", "絲": "丝", "並": "并", "麼": "么",
    "關": "关", "係": "系", "說": "说", "時": "时", "過": "过",
    "對": "对", "會": "会", "動": "动", "體": "体", "現": "现",
    "電": "电", "壓": "压", "機": "机", "線": "线", "路": "路",
    "開": "开", "網": "网", "碼": "码", "號": "号",
    "據": "据", "數": "数", "術": "术", "學": "学", "習": "习",
    "實": "实", "驗": "验", "檢": "检", "測": "测",
    "設": "设", "備": "备", "裝": "装", "傳": "传",
    "輸": "输", "訊": "讯", "連": "连", "接": "接", "頭": "头",
    "纜": "缆", "綫": "线",
    "圖": "图", "書": "书", "畫": "画", "寫": "写", "讀": "读",
    "質": "质", "量": "量", "標": "标", "準": "准", "規": "规",
    "範": "范", "圍": "围", "際": "际", "國": "国", "産": "产",
    "無": "无", "爲": "为", "從": "从", "來": "来", "後": "后",
    "應": "应", "將": "将", "當": "当", "總": "总", "組": "组",
    "處": "处", "發": "发", "進": "进", "車": "车", "門": "门",
    "問": "问", "題": "题", "統": "统", "計": "计", "編": "编",
    "錄": "录", "節": "节", "點": "点",
}

# opencc：优先使用完整繁简转换，不可用时降级为轻量映射
_OPENCC_AVAILABLE = False
_OPENCC_CONVERTER: Any = None
try:
    import opencc  # type: ignore[import-untyped]
    _OPENCC_CONVERTER = opencc.OpenCC("t2s.json")
    _OPENCC_AVAILABLE = True
except (ImportError, Exception):
    pass


class TextCleanNode(BaseNode):
    """
    文档级文本清洗节点

    负责第1-3层 + 第6层，作用于 md_content（切分前的完整文档）。
    第4-5层及质量标记由 chunk_clean_node 在切分后对 chunks 列表执行。
    """

    name = "text_clean_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        对 state['md_content'] 执行 L1-L3 + L6 清洗后写回。

        注意：此节点不处理 chunks 列表。chunks 的清洗由 chunk_clean_node 负责。
        """
        md_content = state.get("md_content", "")
        if not md_content:
            self.logger.info("md_content 为空，跳过文档级清洗")
            return state

        original_len = len(md_content)

        # ---- 第1层：编码层 ----
        if getattr(self.config, "clean_layer1_enabled", True):
            md_content = self._layer1_encoding(md_content)

        # ---- 第2层：空白层 ----
        if getattr(self.config, "clean_layer2_enabled", True):
            md_content = self._layer2_whitespace(md_content)

        # ---- 第3层：结构噪声层 ----
        if getattr(self.config, "clean_layer3_enabled", True):
            md_content = self._layer3_structural_noise(md_content)

        # ---- 第6层：术语归一化 ----
        if getattr(self.config, "clean_layer6_enabled", True):
            md_content = self._layer6_terminology(md_content)

        cleaned_len = len(md_content)
        pct = (original_len - cleaned_len) / max(original_len, 1) * 100
        self.logger.info(
            f"文档级清洗完成 | {original_len} → {cleaned_len} 字符 "
            f"| 减少 {original_len - cleaned_len} ({pct:.1f}%)"
        )

        state["md_content"] = md_content
        return state

    # ========================================================================
    # 第1层：编码层
    # ========================================================================

    def _layer1_encoding(self, text: str) -> str:
        stats: Dict[str, int] = {}

        if getattr(self.config, "clean_normalize_newlines", True):
            before = len(text)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            stats["换行符统一"] = before - len(text)

        if getattr(self.config, "clean_remove_control_chars", True):
            text, removed = self._remove_control_chars(text)
            if removed:
                stats["控制字符"] = removed

        if getattr(self.config, "clean_remove_zero_width_chars", True):
            text, removed = self._remove_zero_width_chars(text)
            if removed:
                stats["零宽字符"] = removed

        if stats:
            detail = " | ".join(f"{k}: {v}个" for k, v in stats.items())
            self.logger.info(f"L1 编码层: {detail}")
        else:
            self.logger.info("L1 编码层: 无需清洗")
        return text

    def _remove_control_chars(self, text: str) -> Tuple[str, int]:
        """去除不可打印 ASCII 控制字符，保留 \\n (0x0A) 和 \\t (0x09)。返回 (清洗后文本, 移除数)。"""
        removed = 0
        cleaned = []
        for ch in text:
            cp = ord(ch)
            if cp in (0x0A, 0x09):
                cleaned.append(ch)
            elif cp < 0x20 or cp == 0x7F:
                cleaned.append(" ")
                removed += 1
            else:
                cleaned.append(ch)
        return "".join(cleaned), removed

    def _remove_zero_width_chars(self, text: str) -> Tuple[str, int]:
        """删除零宽字符，NO-BREAK SPACE 替换为普通空格。返回 (清洗后文本, 移除数)。"""
        removed = 0
        cleaned = []
        for ch in text:
            if ch in ZERO_WIDTH_CHARS:
                removed += 1
                if ch == " ":
                    cleaned.append(" ")
                continue
            cleaned.append(ch)
        return "".join(cleaned), removed

    # ========================================================================
    # 第2层：空白层
    # ========================================================================

    def _layer2_whitespace(self, text: str) -> str:
        stats: Dict[str, int] = {}

        if getattr(self.config, "clean_strip_lines", True):
            text = "\n".join(line.strip() for line in text.split("\n"))

        if getattr(self.config, "clean_merge_blank_lines", True):
            max_blank = getattr(self.config, "clean_max_blank_lines", 2)
            text, compressed = self._merge_blank_lines(text, max_blank)
            if compressed:
                stats["压缩空行组"] = compressed

        if getattr(self.config, "clean_merge_broken_lines", True):
            end_chars = getattr(self.config, "clean_sentence_end_chars",
                                "。！？…」》）)\"'""'：；、")
            use_protection = getattr(self.config, "clean_merge_protected_patterns", True)
            text, merged = self._merge_broken_lines(text, end_chars, use_protection)
            if merged:
                stats["合并断行"] = merged

        if stats:
            detail = " | ".join(f"{k}: {v}处" for k, v in stats.items())
            self.logger.info(f"L2 空白层: {detail}")
        else:
            self.logger.info("L2 空白层: 无需清洗")
        return text

    def _merge_blank_lines(self, text: str, max_blank: int = 2) -> Tuple[str, int]:
        """压缩多余空行，返回 (清洗后文本, 压缩的空行组数)。"""
        if max_blank < 0:
            max_blank = 0
        threshold = max_blank + 2
        replacement = "\n" * (max_blank + 1)
        # 计数：找到连续 max_blank+2 个以上换行符的组数
        pattern = re.compile(r"\n{" + str(threshold) + r",}")
        compressed = len(pattern.findall(text))
        text = pattern.sub(replacement, text)
        return text.lstrip("\n"), compressed

    def _merge_broken_lines(self, text: str, end_chars: str,
                             use_protection: bool = True) -> Tuple[str, int]:
        """
        合并段落内部错误断行。保护结构化内容不被破坏。
        """
        lines = text.split("\n")
        if len(lines) < 2:
            return text, 0

        sentence_ends = set(end_chars)
        merged_count = 0

        def _is_protected(line: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return True
            if re.match(r"^#{1,6}\s+", stripped):
                return True
            if re.match(r"^[-*+]\s+", stripped):
                return True
            if re.match(r"^\d+[.)]\s+", stripped):
                return True
            if stripped.startswith("```"):
                return True
            if stripped.startswith("|"):
                return True
            if re.match(r"^[>\s]*>", stripped):
                return True
            if stripped.startswith("---"):
                return True
            return False

        def _ends_with_punctuation(line: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return True
            return stripped[-1] in sentence_ends

        is_special = _is_protected if use_protection else lambda ln: not ln.strip()

        result = []
        buffer = ""

        for line in lines:
            if is_special(line):
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append(line)
            else:
                if buffer:
                    if not _ends_with_punctuation(buffer):
                        buffer += line.strip()
                        merged_count += 1
                    else:
                        result.append(buffer)
                        buffer = line
                else:
                    buffer = line

        if buffer:
            result.append(buffer)

        return "\n".join(result), merged_count

    # ========================================================================
    # 第3层：结构噪声层
    # ========================================================================

    def _layer3_structural_noise(self, text: str) -> str:
        if not getattr(self.config, "clean_remove_headers_footers", True):
            self.logger.info("L3 结构噪声层: 已关闭")
            return text

        max_len = getattr(self.config, "clean_header_footer_max_length", 80)
        min_occ = getattr(self.config, "clean_header_footer_min_occurrences", 2)

        safety_str = getattr(self.config, "clean_hf_safety_keywords",
                             "安全,警告,注意,危险,严禁,必须,重要,须知,警示,紧急")
        safety_keywords = set(kw.strip() for kw in safety_str.split(",") if kw.strip())

        noise_str = getattr(self.config, "clean_hf_noise_keywords",
                            "页码,页眉,页脚,版权所有,版权,http,www,第,共,转第")
        noise_keywords = set(kw.strip() for kw in noise_str.split(",") if kw.strip())

        text, removed_count, pattern_count, protected_count, noise_count = \
            self._remove_headers_footers(
                text, max_len, min_occ, safety_keywords, noise_keywords
            )

        if removed_count:
            self.logger.info(
                f"L3 结构噪声层: 去除页眉/页脚 {pattern_count} 种模式, "
                f"删除 {removed_count} 行 | "
                f"安全保护 {protected_count} 种 | 噪声确认 {noise_count} 种"
            )
        else:
            self.logger.info("L3 结构噪声层: 未检测到页眉/页脚噪声")
        return text

    def _remove_headers_footers(
        self,
        text: str,
        max_len: int = 80,
        min_occ: int = 2,
        safety_keywords=None,
        noise_keywords=None,
    ) -> Tuple[str, int, int, int, int]:
        """
        频率+关键词双重判定去页眉页脚。

        Returns: (清洗后文本, 删除行数, 噪声模式数, 安全保护数, 噪声确认数)
        """
        if safety_keywords is None:
            safety_keywords = DEFAULT_SAFETY_KEYWORDS
        if noise_keywords is None:
            noise_keywords = DEFAULT_NOISE_KEYWORDS

        lines = text.split("\n")
        if not lines:
            return text, 0, 0, 0, 0

        short_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < max_len:
                short_lines.append(stripped)

        if not short_lines:
            return text, 0, 0, 0, 0

        counter = Counter(short_lines)
        protected_count = 0
        noise_confirmed_count = 0
        header_footer_patterns: Set[str] = set()

        for line, count in counter.items():
            if count < min_occ:
                continue

            if any(kw in line for kw in safety_keywords):
                protected_count += 1
                self.logger.debug(f"安全保护: 跳过重复行「{line[:50]}」(出现{count}次)")
                continue

            if any(kw in line for kw in noise_keywords) or count >= min_occ:
                header_footer_patterns.add(line)
                if any(kw in line for kw in noise_keywords):
                    noise_confirmed_count += 1

        pattern_count = len(header_footer_patterns)
        removed_count = 0

        if header_footer_patterns:
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped in header_footer_patterns:
                    removed_count += 1
                    continue
                cleaned_lines.append(line)
            text = "\n".join(cleaned_lines)

        return text, removed_count, pattern_count, protected_count, noise_confirmed_count

    # ========================================================================
    # 第6层：术语归一化
    # ========================================================================

    def _layer6_terminology(self, text: str) -> str:
        self.log_step("L6", "术语归一化")

        if getattr(self.config, "clean_trad_to_simp", True):
            self.log_step("L6.1", "繁简统一")
            text = self._trad_to_simp(text)

        if getattr(self.config, "clean_fullwidth_to_halfwidth", True):
            self.log_step("L6.2", "全角半角统一")
            text = self._fullwidth_to_halfwidth(text)

        if getattr(self.config, "clean_model_alias_map", False):
            self.log_step("L6.3", "型号别名映射")
            map_path = getattr(self.config, "clean_model_alias_map_path", "")
            text = self._model_alias_replace(text, map_path)

        return text

    def _trad_to_simp(self, text: str) -> str:
        """繁体→简体。优先 opencc，不可用时使用内置轻量映射。"""
        if _OPENCC_AVAILABLE and _OPENCC_CONVERTER:
            try:
                return _OPENCC_CONVERTER.convert(text)
            except Exception as e:
                self.logger.warning(f"opencc 转换失败，回退到内置映射: {e}")

        converted = []
        change_count = 0
        for ch in text:
            simp = _TRAD_TO_SIMP_MAP.get(ch)
            if simp:
                converted.append(simp)
                change_count += 1
            else:
                converted.append(ch)
        if change_count:
            self.logger.info(f"L6 繁简转换: {change_count} 个字符")
        return "".join(converted)

    def _fullwidth_to_halfwidth(self, text: str) -> str:
        """
        全角→半角。

        利用 Unicode 编码规律：全角字母数字 U+FF01-U+FF5E 与 ASCII U+21-U+7E
        偏移量为 0xFEE0。全角空格 U+3000 → 半角空格 U+0020。
        """
        converted = []
        change_count = 0
        for ch in text:
            cp = ord(ch)
            if 0xFF01 <= cp <= 0xFF5E:
                converted.append(chr(cp - 0xFEE0))
                change_count += 1
            elif cp == 0x3000:
                converted.append(" ")
                change_count += 1
            else:
                converted.append(ch)
        if change_count:
            self.logger.info(f"L6 全角→半角: {change_count} 个字符")
        return "".join(converted)

    def _model_alias_replace(self, text: str, map_path: str) -> str:
        """
        型号别名映射（JSON 配置文件驱动）。

        映射文件格式：{"RS-12": ["RS12", "RS 12", "RS_12"], ...}
        标准名 → 别名列表，遇到别名替换为标准名。
        """
        if not map_path:
            return text

        alias_map: Dict[str, List[str]] = {}
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                alias_map = json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"型号别名映射文件不存在: {map_path}")
            return text
        except json.JSONDecodeError as e:
            self.logger.error(f"型号别名映射文件 JSON 解析失败: {e}")
            return text

        replaced = 0
        for canonical, aliases in alias_map.items():
            for alias in aliases:
                if alias in text:
                    text = text.replace(alias, canonical)
                    replaced += 1

        if replaced:
            self.logger.info(f"型号别名映射: {replaced} 处替换")
        return text


# ============================================================================
# 独立测试入口
# ============================================================================

if __name__ == "__main__":
    setup_logging()
    logging.getLogger().setLevel(logging.DEBUG)

    dirty_md = (
        "# 安全操作手册\r\n"
        "\r\n\r\n\r\n\r\n"
        "这是一段被错误换行\n的文本内容，应该被合\n并成一行。\n"
        "\n"
        "## 第二节\x00\x01注意事项\n"
        "​​"
        "## 警告：高压危险\n"
        "\n"
        "ＡＢＣ-123 型号说明\n"
        "聯繫電話：13812345678\n"
        "邮箱：test@example.com\n"
        "\n"
        "第 1 页\n"
        "正常段落内容，含有多余的空白。  \n"
        "\n"
        "# 警告：高压危险\n"
        "\n"
        "另一个段落。\n"
        "\n"
        "第 1 页\n"
        "\n"
        "第三个段落内容。\n"
        "\n"
        "第 1 页\n"
        "\n"
        "### 第三章\n"
        "\n"
        "結尾段落。\n"
    )

    node = TextCleanNode()
    result = node.process({"md_content": dirty_md})

    print("=" * 60)
    print("【文档级清洗结果（L1-L3 + L6）】")
    print(result["md_content"])
    print("=" * 60)
