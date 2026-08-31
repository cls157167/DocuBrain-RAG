"""
稀疏向量 L2 归一化工具

用于 Milvus 存储前的向量预处理，适配 COSINE 距离计算
"""

import math
from typing import Dict, List


def normalize_sparse_l2(sparse_dict: Dict[int, float]) -> Dict[int, float]:
    """
    稀疏向量 L2 归一化

    Args:
        sparse_dict: {token_id: weight} 格式的稀疏向量

    Returns:
        归一化后的稀疏向量，向量长度为 1

    Example:
        >>> normalize_sparse_l2({1: 3.0, 2: 4.0})
        {1: 0.6, 2: 0.8}
    """
    if not sparse_dict:
        return sparse_dict

    norm = math.sqrt(sum(v * v for v in sparse_dict.values()))
    if norm == 0:
        return sparse_dict

    return {k: v / norm for k, v in sparse_dict.items()}


def normalize_sparse_l2_batch(sparse_dicts: List[Dict[int, float]]) -> List[Dict[int, float]]:
    """
    批量 L2 归一化

    Args:
        sparse_dicts: 稀疏向量列表

    Returns:
        归一化后的稀疏向量列表
    """
    return [normalize_sparse_l2(vec) for vec in sparse_dicts]
