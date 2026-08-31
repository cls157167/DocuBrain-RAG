import logging
import os.path
import shutil
import uuid
from datetime import datetime

from fastapi import UploadFile

from knowledge.front.service.task_service import TaskService
from knowledge.front.utils.paths import get_local_base_dir
from knowledge.processor.import_process.import_graph import create_import_graph
from knowledge.processor.import_process.state import ImportGraphState


class ImportFileService:
    logger=logging.Logger(__name__)
    #初始化任务节点
    def __init__(self,task_service:TaskService):
        self.task_service=task_service


    #获取文件保存路径
    def get_pdf_save_path(self):
        return os.path.join(get_local_base_dir(),datetime.now().strftime("%Y%m%d"))

    #保存上传的文件到本地
    def save_upload_file_to_local(self,file:UploadFile,file_dir:str):
        #判断路径是否存在
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        #保存文件
        import_file_path=os.path.join(file_dir,file.filename)
        with open(import_file_path,"wb") as f:
            shutil.copyfileobj(file.file,f)
        return import_file_path


    #调用保存上传的文件到本地的方法，并记录
    def file_upload_process(self,file:UploadFile):
        #获取文件保存路径
        date_dir=self.get_pdf_save_path()
        #生成UUID
        task_id=uuid.uuid4().hex[:12]
        #生成最终路径
        file_path=os.path.join(date_dir,task_id)


        #记录当前正在运行的节点，前端使用
        self.task_service.mark_node_running(task_id,"upload_file")
        #调用方法，将文件保存到本地
        import_file_path=self.save_upload_file_to_local(file,file_path)
        #记录当前完成节点，前端使用
        self.task_service.mark_node_done(task_id,"upload_file")
        return task_id,file_path,import_file_path

    def run_upload_file_task(self,task_id,file_path,import_file_path):

        try:
            #将节点状态更新为运行中
            self.task_service.update_task_status(task_id,"processing")

            #调用create_import_graph方法构建图对象，开始运行各个节点
            #构建初始数据
            initial_status:ImportGraphState={
                "task_id": task_id,
                "import_file_path":import_file_path,
                "file_dir":file_path
            }
            #构建图实例
            agent=create_import_graph()

            for event in agent.stream(initial_status):
                for node_name,node_data in event.items():
                    self.task_service.mark_node_done(task_id, node_name)
                    self.logger.info(f"{task_id}完成了节点{node_name}")

            #标记任务处理完成
            self.task_service.update_task_status(task_id,"completed")
        except Exception as e:
            self.task_service.update_task_status(task_id, "failed")
            self.logger.exception(f"文件向量化任务{task_id}失败")
            raise e

