from langgraph.graph import StateGraph,END

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.chunks_embedding_node import ChunksEmbeddingNode
from knowledge.processor.import_process.nodes.chunks_save_to_milvus_node import ChunksSaveToMilvusNode
from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.md_Img_node import MDImgNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PDFToMDNode
from knowledge.processor.import_process.nodes.chunk_clean_node import ChunkCleanNode
from knowledge.processor.import_process.nodes.text_clean_node import TextCleanNode
from knowledge.processor.import_process.nodes.kg_build_node import KGBuildNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def route_after_entry(state:ImportGraphState):
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    if state.get("is_md_read_enabled"):
        return "md_Img_node"
    return END

def create_import_graph():
    """创建导入流程图"""
    #创建工作流
    workflow=StateGraph(ImportGraphState)

    #实例化节点
    nodes={
        "entry_node":EntryNode(),
        "pdf_to_md_node":PDFToMDNode(),
        "md_Img_node":MDImgNode(),
        "document_split_node":DocumentSplitNode(),
        "chunk_clean_node":ChunkCleanNode(),
        "item_name_recognition_node":ItemNameRecognitionNode(),
        "text_clean_node":TextCleanNode(),
        "chunks_embedding_node":ChunksEmbeddingNode(),
        "chunks_save_to_milvus_node":ChunksSaveToMilvusNode(),
        "kg_build_node":KGBuildNode()
    }

    #添加节点
    for name,node in nodes.items():
        workflow.add_node(name,node)

    #添加入口节点，条件边，顺序边
    workflow.set_entry_point("entry_node")

    workflow.add_conditional_edges(
        "entry_node",
        route_after_entry,
        {
            "pdf_to_md_node":"pdf_to_md_node",
            "md_Img_node":"md_Img_node",
            END:END
        }
    )
    workflow.add_edge("pdf_to_md_node","md_Img_node")
    workflow.add_edge("md_Img_node","text_clean_node")
    workflow.add_edge("text_clean_node","document_split_node")
    workflow.add_edge("document_split_node","chunk_clean_node")
    workflow.add_edge("chunk_clean_node","item_name_recognition_node")
    workflow.add_edge("item_name_recognition_node","chunks_embedding_node")
    workflow.add_edge("chunks_embedding_node","chunks_save_to_milvus_node")
    workflow.add_edge("chunks_save_to_milvus_node","kg_build_node")
    workflow.add_edge("kg_build_node",END)

    #编译工作流
    workflow_compile=workflow.compile()
    return workflow_compile

def run_import(file_dir:str,import_file_path:str):

    #构建初始状态
    initial_state=create_default_state(file_dir=file_dir,import_file_path=import_file_path)
    final_state=None

    #创建图实例
    agent=create_import_graph()

    for event in agent.stream(initial_state):
        for node_name,node_state in event.items():
            print(f"运行节点：{node_name}")
            print(f"节点数据：{node_state}")
            final_state=node_state

    return final_state or initial_state

# ==================== 命令行入口 ====================

if __name__ == "__main__":
    # 1. 配置日志
    setup_logging()

    print("=" * 50)
    print("知识库导入流程测试")
    print("=" * 50)

    # 2. 准备测试文件路径
    # 请根据实际情况修改以下路径
    test_file_dir = r"D:\$AAA\4、Large Models\项目\项目1\2.资料\2-pdf文档\pdf文档\doc"
    test_import_file_path = r"D:\$AAA\4、Large Models\项目\项目1\2.资料\2-pdf文档\pdf文档\doc\万用表RS-12的使用.pdf"

    # 检查文件是否存在
    from pathlib import Path
    test_path = Path(test_import_file_path)
    if not test_path.exists():
        print(f"错误: 测试文件不存在: {test_import_file_path}")
        print("请修改 test_import_file_path 为有效的 PDF 或 MD 文件路径")
        exit(1)

    print(f"输入文件: {test_path.name}")
    print(f"文件类型: {test_path.suffix}")
    print("-" * 50)

    # 3. 运行导入流程
    try:
        result = run_import(test_file_dir, test_import_file_path)

        print("-" * 50)
        print("流程完成!")
        print(f"识别商品: {result.get('item_name', 'N/A')}")
        print(f"切片数量: {len(result.get('chunks', []))}")

    except Exception as e:
        print(f"流程执行失败: {e}")
        import traceback
        traceback.print_exc()

    # 4. 打印图结构（ASCII 可视化）
    print("-" * 50)
    print("图结构:")
    create_import_graph().get_graph().print_ascii()