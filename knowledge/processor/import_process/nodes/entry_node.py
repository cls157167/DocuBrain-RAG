from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState




class EntryNode(BaseNode):
    name = "entry_node"
    #实现process类
    def process(self,state:ImportGraphState):
        """
        处理入口逻辑
        1、获取页面文件的路径
        2、通过后缀名检测文件类型
        3、根据文件类型决定是否启用相应的文件读取
        4、提取文件标题（不含拓展名）
        :param state:图状态
        :return:更新后的图状态
        """
        #1、获取页面文件路径
        self.log_step("step_1", "获取文件路径")
        import_file_path=state["import_file_path"]

        #判断文件路径是否存在
        if not import_file_path:
            raise ValidationError("import_file_path 不能为空",node_name=self.name)


        #2、通过文件路径获取后缀名
        #获取path对象
        file_path=Path(import_file_path)
        #通过path对象获取后缀名
        suffix=file_path.suffix.lower()
        self.log_step("step_2",f"已检测到文件类型为：{suffix}")

        #3、判断文件类型，并启用想用的文件读取
        if suffix==".md":
            state["is_md_read_enabled"]=True
            state["md_path"]=import_file_path
            self.logger.info("启用md文件读取")

        elif suffix==".pdf":
            state["is_pdf_read_enabled"]=True
            state["pdf_path"]=import_file_path
            self.logger.info("启用pdf文件读取")
        else:
            self.logger.warning(f"不支持的文件类型：{suffix}")

        #4、读取文件标题
        state["file_title"]=file_path.stem
        self.logger.info(f"读取到文件标题：{state['file_title']}")

        return state


if __name__=="__main__":
    setup_logging()
    #构建状态
    state={
        "import_file_path":r"D:\$AAA\4、Large Models\项目\项目1\2.资料\2-pdf文档\pdf文档\doc\6W100-整本手册.pdf"
        # "import_file_path":""
    }

    entry_node=EntryNode()

    resp=entry_node.process(state)
    print(resp)