import os.path

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.params import Depends
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from knowledge.front.schema.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.front.service.query_service import QueryService
from knowledge.front.utils.deps import get_query_service
from knowledge.front.utils.paths import get_front_page_dir
from knowledge.processor.query_process.base import setup_logging
from knowledge.utils.sse_util import sse_generator


def register_router(app):

    #访问前端页面
    @app.get("/chat")
    async def get_chat():
        return FileResponse(os.path.join(get_front_page_dir(),"chat.html"))

    @app.post("/query")
    async def query(request:QueryRequest,
                    background_tasks:BackgroundTasks,
                    service:QueryService=Depends(get_query_service)):
        """处理查询请求"""
        #获取基础数据
        user_query=request.query
        session_id=request.session_id or service.generate_session_id()
        is_stream=request.is_stream

        #更新状态，如果是流式输出，建立SSE消息队列
        task_id=service.generate_task_id()
        service.submit_query(task_id,is_stream)

        #根据模式选择处理方式
        if is_stream:
            background_tasks.add_task(
                service.run_query_graph,task_id,session_id,user_query,is_stream)
            return StreamSubmitResponse(
                message="查询任务已提交",
                session_id=session_id,
                task_id=task_id
            )
        else:
            service.run_query_graph(task_id,session_id,user_query,is_stream)
            answer =service.get_answer(task_id)
            return QueryResponse(
                message="处理完成",
                session_id=session_id,
                answer=answer
            )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        return StreamingResponse(
            sse_generator(task_id, request), media_type="text/event-stream",
        )

    @app.get("/history/{session_id}")
    async def get_history(
        session_id: str, limit: int = 50,
        service:QueryService=Depends(get_query_service),
    ):
        try:
            items=service.get_history(session_id,limit)
            return {"session_id":session_id,"items":items}
        except Exception as e:
            raise HTTPException(status_code=500,detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id:str,
            service:QueryService=Depends(get_query_service)
    ):
        count=service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}

def create_app()->FastAPI:
    setup_logging()
    #创建app实例
    app=FastAPI(
        title="Query Service",
        description="知识库查询系统"
    )

    # 允许跨域（开发环境）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,  # 允许携带凭证
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头部
    )

    #静态文件挂载
    page_dir = get_front_page_dir()
    if os.path.exists(page_dir):
        app.mount("/queryfront", StaticFiles(directory=page_dir), name="front")


    #注册路由
    register_router(app)

    return app

app=create_app()

if __name__=="__main__":
    uvicorn.run(app=app,host="0.0.0.0",port=8001)
