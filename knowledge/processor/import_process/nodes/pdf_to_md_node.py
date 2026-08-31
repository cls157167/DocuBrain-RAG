import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Tuple
import logging
from dns import name

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PDFToMDNode(BaseNode):
    name = "pdf_to_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        将PDF转换成MD格式
        第一步：获取PDF文件路径，获取MD文件导出路径，并判断是否为空
        第二步：执行_execute_mineru()方法，将PDF转换成MD
        第三步：返回转换后的MD文件路径
        :param state:
        :return:
        """
        #获取PDF路径与转出MD路径
        pdf_path_obj,out_md_path_obj=self._validate_paths(state)

        #执行PDF转换逻辑
        return_code=self._execute_mineru(pdf_path_obj, out_md_path_obj)
        if return_code != 0:
            raise PdfConversionError("PDF转换失败，检查[mineru]日志后重试",node_name=self.name)

        #获取md_path
        md_path=self.get_md_path(pdf_path_obj, out_md_path_obj)
        state["md_path"]=md_path
        self.log_step("step3",f"返回{state['md_path']}")
        return state

    def _validate_paths(self, state)->Tuple:

        self.log_step("step_1.1", "获取PDF文件路径,并判断是否为空")
        pdf_path = state.get("pdf_path", "")
        if not pdf_path:
            raise FileProcessingError("pdf文件路径不存在", node_name=self.name)
        self.logger.info(f"已获取PDF文件路径{pdf_path},并将其转换为Path实例")
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.is_file():
            raise FileProcessingError("pdf文件不存在", node_name=self.name)

        self.log_step("step1.2", "获取导出的MD文件路径，不存在就用PDF文件的父目录")
        out_md_path = state.get("file_dir", "")
        if not out_md_path:
            out_md_path = str(pdf_path_obj.parent)
        out_md_path_obj = Path(out_md_path)
        self.logger.info(f"已获取导出MD文件的路径{out_md_path}")

        self.log_step("step1.3", f"返回{pdf_path_obj}和{out_md_path_obj}")
        return pdf_path_obj, out_md_path_obj

    def _execute_mineru(self,pdf_path_obj:Path,out_md_path_obj:Path)->int:

        self.log_step("step2.1","构建命令")
        cmd=[
            "mineru",
            "-p",str(pdf_path_obj),
            "-o",str(out_md_path_obj),
            "--source", "local",
            "--device", "cpu",
            "--backend","pipeline",
            "--batch-size", "1"
        ]
        start_ts=time.time()

        self.log_step("step2.2","调用命令行工具")
        proc=subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
            bufsize=1,
        )

        self.log_step("step2.3","MinerU 开始执行,实时输出日志。。。。。")
        for line in proc.stdout:
            self.logger.debug(f"[mineru]{line.rstrip()}")


        return_code=proc.wait()
        elapsed=time.time()-start_ts

        if return_code==0:
            self.logger.info(f"PDF转换成功,耗时{elapsed}秒")
        else:
            self.logger.error("PDF转换失败")

        return return_code

    def get_md_path(self,pdf_path_obj:Path,out_md_path_obj:Path):

        md_file_name=pdf_path_obj.stem
        md_path=out_md_path_obj/md_file_name/"auto"/f"{md_file_name}.md"
        return str(md_path)


if __name__=="__main__":

    setup_logging()
    logger=logging.getLogger(__name__)

    state={
        # "pdf_path":r"D:\$AAA\4、Large Models\项目\项目1\2.资料\2-pdf文档\pdf文档\doc\6W100-整本手册.pdf"
        "pdf_path":""
    }

    try:
        PDFToMDNode=PDFToMDNode()
        PDFToMDNode.process(state)
    except Exception as e:
        logger.exception(f"运行发生异常，异常原因是{e}")

