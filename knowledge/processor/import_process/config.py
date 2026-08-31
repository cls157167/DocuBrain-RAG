"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import Set, Optional
import os
from dotenv import load_dotenv
load_dotenv()

@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000  # 切片最大长度
    min_content_length: int = 500   # 合并短内容的最小长度
    overlap_sentences: int = 1      # 句子级切分时的重叠句数
    item_name_chunk_k: int = 3      # 商品名识别时使用的切片数量
    item_name_chunk_size: int=500

    # ==================== 文本清洗配置（6层架构） ====================
    # 每个步骤有独立开关 + 可调参数，方便按文档类型（说明书/合同/新闻）定制
    #
    # 六层架构：
    #   第1层 编码层 — 统一换行符、删控制字符和零宽字符（基本不会误杀）
    #   第2层 空白层 — 压缩连续空行、去行首尾空白、合并错误断行（保护结构化内容）
    #   第3层 结构噪声层 — 去页眉页脚/页码/导航文本（保护安全警告类重复内容）
    #   第4层 去重层 — 精确去重(MD5) + 模糊去重(默认关闭，只标记不删，rerank降权)
    #   第5层 脱敏层 — 手机号/邮箱/身份证正则掩码，失败写入隔离表
    #   第6层 术语归一化 — 繁简统一、全角半角统一、型号别名映射

    # ============ 第1层：编码层 ============
    clean_layer1_enabled: bool = True           # 编码层总开关
    clean_normalize_newlines: bool = True       # 统一换行符 \r\n → \n
    clean_remove_control_chars: bool = True     # 去除不可打印控制字符 (\x00-\x08, \x0b-\x1f, \x7f)
    clean_remove_zero_width_chars: bool = True  # 去除零宽字符 (​, ‌, ‍, ﻿, ­ 等)

    # ============ 第2层：空白层 ============
    clean_layer2_enabled: bool = True           # 空白层总开关
    clean_strip_lines: bool = True              # 去除行首尾空白
    clean_merge_blank_lines: bool = True        # 合并连续多余空行
    clean_merge_broken_lines: bool = True       # 合并段落内部错误断行
    clean_max_blank_lines: int = 2              # 连续空行最多保留行数
    # 句末标点集：以这些字符结尾的行不合并下一行
    clean_sentence_end_chars: str = "。！？…」》）)\"'""'：；、"
    # 断行合并保护规则：这些结构不参与合并（表格行、列表项、代码块、引用块、分隔线）
    clean_merge_protected_patterns: bool = True  # 启用结构化内容保护

    # ============ 第3层：结构噪声层 ============
    clean_layer3_enabled: bool = True           # 结构噪声层总开关
    clean_remove_headers_footers: bool = True   # 去除重复页眉/页脚/页码
    clean_header_footer_max_length: int = 80    # 候选页眉行最大字符数
    clean_header_footer_min_occurrences: int = 2 # 最小出现次数判定为噪声（2-4可调）
    # 安全关键词：包含这些词的行即使重复也不删除（安全警告类重要内容保护）
    clean_hf_safety_keywords: str = "安全,警告,注意,危险,严禁,必须,重要,须知,警示,紧急"
    # 页眉页脚噪声关键词：含这些词的低信息量重复行优先判定为噪声
    clean_hf_noise_keywords: str = "页码,页眉,页脚,版权所有,版权,http,www,第,共,转第"

    # ============ 第4层：去重层 ============
    clean_layer4_enabled: bool = True           # 去重层总开关
    clean_exact_dedup: bool = True              # 精确去重（MD5），直接删除完全重复chunk
    clean_fuzzy_dedup: bool = False             # 模糊去重，默认关闭，只标记不删除
    clean_fuzzy_dedup_threshold: float = 0.85   # 模糊去重相似度阈值（0-1）
    clean_fuzzy_dedup_action: str = "mark"      # 模糊去重动作: "mark"(标记降权) / "delete"(删除)

    # ============ 第5层：脱敏层 ============
    clean_layer5_enabled: bool = True           # 脱敏层总开关
    clean_desensitize: bool = True              # 是否执行脱敏
    clean_desensitize_phone: bool = True        # 手机号掩码 (保留前3后4)
    clean_desensitize_email: bool = True        # 邮箱掩码 (保留首字符和域名)
    clean_desensitize_id_card: bool = True      # 身份证号掩码 (保留前6后4)
    clean_desensitize_fail_action: str = "isolate"  # 脱敏失败处理: "isolate"(写隔离表) / "skip"(跳过该chunk) / "warn"(仅告警)
    clean_desensitize_isolation_enabled: bool = False  # 是否启用脱敏失败隔离表

    # ============ 第6层：术语归一化 ============
    clean_layer6_enabled: bool = True           # 术语归一化总开关
    clean_trad_to_simp: bool = True             # 繁体中文 → 简体中文
    clean_fullwidth_to_halfwidth: bool = True   # 全角字符 → 半角字符（字母数字符号）
    clean_model_alias_map: bool = False         # 型号别名映射（需配置映射表，默认关闭）
    # 自定义型号别名映射 JSON 文件路径（为空则跳过）
    clean_model_alias_map_path: str = ""

    # ============ 质量标记 ============
    clean_min_chunk_length: int = 10            # 短chunk阈值（字符数），低于此值标记为 too_short
    clean_empty_chunk_action: str = "drop"      # 空chunk处理: "drop"(静默丢弃) / "raise"(抛异常) / "keep"(保留空字符串)

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    """
    简单默认值（name: str = "default"）用于不可变类型；
    field(default_factory=list)用于可变类型，避免多个实例共享同一个对象；
    field(default_factory=lambda: os.getenv(...))用于从环境变量动态读取配置，确保每次创建实例时都获取最新的环境变量值。
    使用 default_factory的关键原因是：简单默认值在类定义时计算一次，而 default_factory在每次实例化时计算，这对于可变类型和环境变量读取至关重要。"
    """
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== Neo4j 配置 ====================
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 5

    # ==================== 速率限制 ====================
    requests_per_minute: int = 12  # 图片总结 API 速率限制

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()



# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config     # 告诉 Python：我要用的是外面的那个全局变量
    if _config is None:
        _config = ImportConfig.from_env()  # 修改的是全局变量
    return _config



