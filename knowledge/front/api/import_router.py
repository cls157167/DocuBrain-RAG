import os.path
from logging import info
from pathlib import Path

import uvicorn
from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from fastapi.params import Depends
from fastapi.middleware.cors import CORSMiddleware

from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse

from knowledge.front.schema.task_schema import TaskStatusResponse
from knowledge.front.schema.upload_schema import UploadResponse
from knowledge.front.service.import_file_service import ImportFileService
from knowledge.front.service.task_service import TaskService
from knowledge.front.utils.deps import get_import_file_service, get_task_service
from knowledge.front.utils.paths import FRONT_PAGE_DIR, get_front_page_dir
from knowledge.processor.import_process.base import setup_logging


def _regist_router(app:FastAPI):

    #页面接口
    @app.get("/import")
    async def import_page():
        return FileResponse(path=os.path.join(get_front_page_dir(),"import.html"))

    #上传接口
    #后台任务对象
    #上传文件对象
    #处理文件的逻辑
    @app.post("/upload",response_model=UploadResponse)
    async def upload_file(
            background_tasks:BackgroundTasks,
            file:UploadFile=File(...),
            service:ImportFileService=Depends(get_import_file_service))->UploadResponse:

        #把上传的文件，保存到本地路径
        task_id,file_path,import_file_path=service.file_upload_process(file)

        #创建后台任务，执行向量化的各个节点
        background_tasks.add_task(service.run_upload_file_task,task_id,file_path,import_file_path)

        #返回结果id
        return UploadResponse(message="向量化成功",task_id=task_id)


    #根据task_id查询节点进度的接口
    @app.get("/status/{task_id}",response_model=TaskStatusResponse)
    async def status_endpoints(task_id,service:TaskService=Depends(get_task_service))-> TaskStatusResponse:
        task_info=service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)


def create_app():
    setup_logging()

    app=FastAPI(
        title="知识库文档导入系统",
        description="上传文档，自动向量化并保存到milvus向量数据库中",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #挂在静态资源
    static_dir=os.path.join(FRONT_PAGE_DIR,"front")
    if os.path.exists(static_dir):
        app.mount("/front",StaticFiles(directory=static_dir),name="static")

    #注册路由
    _regist_router(app)

    return app

if __name__=="__main__":
    uvicorn.run(
        app="knowledge.front.api.import_router:create_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        factory = True
    )

