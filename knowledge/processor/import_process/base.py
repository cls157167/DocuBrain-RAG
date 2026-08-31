"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()
        module = self.__class__.__module__
        cls_name = self.__class__.__qualname__
        # 当模块是 __main__ 时，用类名代替
        logger_name = cls_name if module == "__main__" else f"{module}.{cls_name}"
        self.logger = logging.getLogger(logger_name)

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """

        """
        "__init__和 __call__是 Python 中两个不同的魔术方法。__init__在创建实例时自动执行一次，用于初始化对象的配置、日志、数据库连接等资源。
        __call__不会自动执行，只有在把实例当作函数调用时才会触发，每次调用都会执行。
        在 LangGraph 中，__init__负责节点的一次性初始化，__call__负责每次图执行时的节点逻辑，两者分工明确。"
        """

        """
        BaseNode 类定义了 __call__（第52行），所以当你这样做时：
        
        node = MyNode()
        workflow.add_node("my_node", node)  # 把实例直接传给 LangGraph
        
        或者workflow.add_node("my_node", MyNode())
        
        LangGraph 内部会像调用函数一样调用这个实例：node(state)，实际触发的就是 __call__ 方法，然后 __call__ 里再调用 self.process(state) 执行具体逻辑。

        """

        self.logger.info(f"--- {self.name} 开始 ---")


        try:
            result = self.process(state)
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except ImportProcessError:
            # 已经是自定义异常，直接抛出
            raise
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志

        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)



# 配置日志格式
def setup_logging(level: int = logging.DEBUG):
    root_logger = logging.getLogger()

    # 防止重复添加 handler
    if root_logger.handlers:
        return

    root_logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s -  %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(r'D:\$AAA\4、Large Models\项目\项目1\日志\app.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
