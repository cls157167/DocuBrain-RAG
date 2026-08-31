"""
Neo4j 客户端工具

封装 Neo4j 连接管理和常用 Cypher 操作，为 KG 构建和检索提供统一接口。
"""

import logging
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver, Session

load_dotenv()
logger = logging.getLogger(__name__)

# ---- 连接管理 ----

_driver: Optional[Driver] = None


def get_neo4j_driver() -> Optional[Driver]:
    """获取 Neo4j 驱动单例。未配置时返回 None 而非抛异常。"""
    global _driver
    if _driver is not None:
        return _driver

    uri = os.getenv("NEO4J_URI", "")
    username = os.getenv("NEO4J_USERNAME", "")
    password = os.getenv("NEO4J_PASSWORD", "")

    if not uri:
        logger.warning("NEO4J_URI 未配置，KG 功能不可用")
        return None

    try:
        _driver = GraphDatabase.driver(uri, auth=(username, password))
        _driver.verify_connectivity()
        logger.info(f"Neo4j 连接成功: {uri}")
        return _driver
    except Exception as e:
        logger.warning(f"Neo4j 连接失败: {e}，KG 功能将跳过")
        _driver = None
        return None


def close_neo4j_driver():
    """关闭 Neo4j 驱动连接。"""
    global _driver
    if _driver:
        _driver.close()
        _driver = None


# ---- Cypher 执行 ----

def run_cypher(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    db: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    执行 Cypher 查询并返回结果列表。

    Args:
        query: Cypher 查询语句
        params: 查询参数
        db: 数据库名，默认从环境变量 NEO4J_DATABASE 读取

    Returns:
        结果记录列表，每条记录为 dict。连接不可用时返回空列表。
    """
    driver = get_neo4j_driver()
    if driver is None:
        return []

    database = db or os.getenv("NEO4J_DATABASE", "neo4j")

    try:
        with driver.session(database=database) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    except Exception as e:
        logger.exception(f"Cypher 执行失败: {e}")
        return []


def run_cypher_batch(
    queries: List[tuple],
    db: Optional[str] = None,
) -> bool:
    """
    批量执行 Cypher 语句（单事务）。

    Args:
        queries: [(cypher_str, params_dict), ...] 列表
        db: 数据库名

    Returns:
        是否全部执行成功
    """
    driver = get_neo4j_driver()
    if driver is None:
        return False

    database = db or os.getenv("NEO4J_DATABASE", "neo4j")

    try:
        with driver.session(database=database) as session:
            with session.begin_transaction() as tx:
                for cypher, params in queries:
                    tx.run(cypher, params or {})
                tx.commit()
        return True
    except Exception as e:
        logger.exception(f"Cypher 批量执行失败: {e}")
        return False


# ---- KG 构建专用操作 ----

def merge_entity(
    entity_name: str,
    entity_type: str,
    chunk_id: int,
    item_name: str = "",
) -> Optional[str]:
    """
    创建或合并实体节点，返回节点 ID。

    使用 MERGE 避免重复创建同名实体。同名实体可能被多个 chunk 引用，
    MERGE 保证唯一性，同时追加 chunk 引用关系。
    """
    cypher = """
    MERGE (e:Entity {name: $name})
    SET e.type = $type,
        e.item_name = $item_name
    RETURN elementId(e) AS entity_id
    """
    results = run_cypher(cypher, {
        "name": entity_name,
        "type": entity_type,
        "item_name": item_name,
    })
    if results:
        return results[0].get("entity_id")
    return None


def merge_relation(
    from_name: str,
    to_name: str,
    rel_type: str,
    description: str = "",
    chunk_id: int = 0,
):
    """
    创建或合并实体间关系。

    使用 MERGE 避免重复创建同类型关系。
    """
    cypher = """
    MATCH (a:Entity {name: $from_name})
    MATCH (b:Entity {name: $to_name})
    MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
    SET r.description = $description,
        r.chunk_id = $chunk_id
    """
    run_cypher(cypher, {
        "from_name": from_name,
        "to_name": to_name,
        "rel_type": rel_type,
        "description": description,
        "chunk_id": chunk_id,
    })


def link_entity_to_chunk(entity_name: str, chunk_id: int):
    """建立实体与 chunk 的引用关系（用于检索时反查 chunk）。"""
    cypher = """
    MATCH (e:Entity {name: $name})
    SET e.chunk_id = coalesce(e.chunk_id, []) + $chunk_id
    """
    run_cypher(cypher, {"name": entity_name, "chunk_id": chunk_id})


# ---- KG 检索专用操作 ----

def search_entities_by_name(keyword: str, limit: int = 10) -> List[Dict]:
    """
    按名称模糊匹配实体（用于实体对齐的第一步：快速召回候选）。

    Args:
        keyword: 实体关键词
        limit: 返回上限

    Returns:
        [{name, type, item_name, chunk_ids}, ...]
    """
    cypher = """
    MATCH (e:Entity)
    WHERE e.name CONTAINS $keyword
    RETURN e.name AS name,
           e.type AS type,
           e.item_name AS item_name,
           e.chunk_id AS chunk_ids
    LIMIT $limit
    """
    return run_cypher(cypher, {"keyword": keyword, "limit": limit})


def traverse_from_entities(
    entity_names: List[str],
    max_hops: int = 2,
    max_triples: int = 50,
) -> Dict[str, Any]:
    """
    从种子实体出发做图遍历，返回关联的子图。

    Args:
        entity_names: 种子实体名称列表
        max_hops: 最大跳数（1-2）
        max_triples: 返回三元组上限

    Returns:
        {
            "triples": [{head, relation, tail, description}, ...],
            "entity_names": [str, ...],
            "chunk_ids": [int, ...]
        }
    """
    # 1-2 跳遍历
    cypher = f"""
    MATCH (seed:Entity)
    WHERE seed.name IN $names
    MATCH path = (seed)-[r:RELATED*1..{max_hops}]-(neighbor:Entity)
    WITH relationships(path) AS rels, neighbor
    UNWIND rels AS r
    WITH DISTINCT startNode(r).name AS head,
                  r.type AS relation,
                  endNode(r).name AS tail,
                  r.description AS description,
                  r.chunk_id AS chunk_id
    RETURN head, relation, tail, description, chunk_id
    LIMIT $max_triples
    """
    results = run_cypher(cypher, {
        "names": entity_names,
        "max_triples": max_triples,
    })

    if not results:
        return {"triples": [], "entity_names": [], "chunk_ids": []}

    triples = [
        {
            "head": r.get("head", ""),
            "relation": r.get("relation", ""),
            "tail": r.get("tail", ""),
            "description": r.get("description", ""),
        }
        for r in results
    ]

    # 收集所有涉及的 chunk_ids
    chunk_ids = []
    for r in results:
        cid = r.get("chunk_id")
        if cid and cid not in chunk_ids:
            chunk_ids.append(cid)

    # 收集所有实体名
    entity_names_set = set()
    for r in results:
        entity_names_set.add(r.get("head", ""))
        entity_names_set.add(r.get("tail", ""))

    return {
        "triples": triples,
        "entity_names": list(entity_names_set),
        "chunk_ids": chunk_ids,
    }


def clear_graph(item_name: Optional[str] = None):
    """
    清空图数据（用于重导入）。指定 item_name 时只删除该产品的实体和关系。

    Args:
        item_name: 产品名，为 None 时清空全部
    """
    if item_name:
        cypher = """
        MATCH (e:Entity {item_name: $item_name})
        DETACH DELETE e
        """
        run_cypher(cypher, {"item_name": item_name})
        logger.info(f"已清空产品 '{item_name}' 的 KG 数据")
    else:
        run_cypher("MATCH (n) DETACH DELETE n")
        logger.info("已清空全部 KG 数据")


# ---- 索引创建（首次连接时调用） ----

def ensure_indexes():
    """创建必要的索引和约束，首次使用 KG 功能时调用。"""
    indexes = [
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "CREATE INDEX entity_item_name IF NOT EXISTS FOR (e:Entity) ON (e.item_name)",
    ]
    for cypher in indexes:
        run_cypher(cypher)
    logger.info("KG 索引检查完成")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试连接
    driver = get_neo4j_driver()
    if driver:
        print("Neo4j 连接测试成功")
        ensure_indexes()

        # 测试 CRUD
        merge_entity("测试设备-X100", "设备", 1, "测试产品")
        merge_entity("测试部件-Y200", "部件", 1, "测试产品")
        merge_relation("测试设备-X100", "测试部件-Y200", "has_part", "包含关系测试", 1)
        link_entity_to_chunk("测试设备-X100", 1)

        # 测试检索
        found = search_entities_by_name("X100")
        print(f"实体搜索: {found}")

        result = traverse_from_entities(["测试设备-X100"])
        print(f"图遍历三元组: {result['triples']}")
        print(f"关联 chunk_ids: {result['chunk_ids']}")

        # 清理测试数据
        clear_graph("测试产品")
        close_neo4j_driver()
    else:
        print("Neo4j 未连接，跳过测试（这是正常的，如果你还没配 NEO4J_URI）")
