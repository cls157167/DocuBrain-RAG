"""
SSE (Server-Sent Events) 工具模块

提供 SSE 事件类型、消息打包、队列管理和异步生成器，
配合 FastAPI StreamingResponse 实现服务器到客户端的实时单向推送。
"""
import json
import queue
import asyncio
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request
from starlette.responses import StreamingResponse


# ============================================================
# 1. SSE 事件类型枚举
# ============================================================

class SSEEvent(str, Enum):
    """
    SSE 标准事件类型
    - READY:  连接建立成功，管道已通
    - MESSAGE: 正常的业务数据推送
    - ERROR:   服务器内部错误
    - DONE:    数据全部推送完毕，前端可关闭连接
    - PING:    心跳，保持连接不断
    """
    READY = "ready"
    MESSAGE = "message"
    PROGRESS = "progress"
    ERROR = "error"
    DONE = "done"
    PING = "ping"


# ============================================================
# 2. SSE 消息打包
# ============================================================

def _sse_pack(event: SSEEvent, data: Any) -> str:
    """
    将事件类型和数据打包为标准的 SSE 文本格式。

    SSE 协议格式:
        event: <事件类型>\\n
        data: <JSON 字符串>\\n\\n

    Args:
        event: SSE 事件类型枚举
        data:  业务数据（任意可 JSON 序列化的对象）

    Returns:
        格式化的 SSE 文本字符串
    """
    # 如果 data 已经是字符串（如错误信息），直接使用；否则 JSON 序列化
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)

    return f"event: {event.value}\ndata: {payload}\n\n"


# ============================================================
# 3. SSE 队列管理
# ============================================================

# 全局字典：task_id → queue.Queue
# 每个 task_id 对应一个线程安全的队列，用于跨协程/线程传递消息
_sse_queue_map: Dict[str, queue.Queue] = {}


def get_or_create_sse_queue(task_id: str) -> queue.Queue:
    """
    获取或创建指定 task_id 对应的 SSE 消息队列。

    队列是线程安全的（queue.Queue），因此既可以在 async 协程中使用
    run_in_executor 安全地读取，也可以在同步的工作线程中直接 put 消息。

    Args:
        task_id: 任务唯一标识

    Returns:
        该任务对应的 queue.Queue 实例
    """
    if task_id not in _sse_queue_map:
        _sse_queue_map[task_id] = queue.Queue()
    return _sse_queue_map[task_id]


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """
    获取指定 task_id 对应的 SSE 消息队列（只读，不创建）。

    Args:
        task_id: 任务唯一标识

    Returns:
        队列实例，如果不存在则返回 None
    """
    return _sse_queue_map.get(task_id, None)


def remove_sse_queue(task_id: str) -> None:
    """
    移除并清理指定 task_id 对应的 SSE 消息队列。

    应在 SSE 连接关闭时调用，防止内存泄漏。

    Args:
        task_id: 任务唯一标识
    """
    _sse_queue_map.pop(task_id, None)


def push_sse_message(task_id: str, event: SSEEvent, data: Any) -> None:
    """
    向指定任务的 SSE 队列推送一条消息。

    供业务代码（如 LangGraph 节点回调、后台任务等）调用，
    将进度、结果等数据推送到 SSE 流中。

    Args:
        task_id: 任务唯一标识
        event:   SSE 事件类型
        data:    要推送的数据

    Raises:
        ValueError: 如果 task_id 对应的队列不存在
    """
    q = get_sse_queue(task_id)
    if q is None:
        raise ValueError(f"task_id='{task_id}' 的 SSE 队列不存在，请先调用 get_or_create_sse_queue 创建")
    q.put({"event": event, "data": data})


# ============================================================
# 4. SSE 异步生成器（核心）
# ============================================================

async def sse_generator(
    task_id: str,
    request: Request,
    *,
    heartbeat: float = 30.0,
    timeout: float = 1.0,
) -> AsyncGenerator[str, None]:
    """
    SSE 异步生成器，用于 FastAPI 的 StreamingResponse。

    工作流程:
        1. 发送 READY 事件 → 通知前端管道就绪
        2. 循环从队列中获取消息 → 打包为 SSE 格式 → yield 推送
        3. 发送 DONE 事件 → 通知前端数据流结束

    设计要点:
        - 使用 run_in_executor 将同步阻塞的 queue.get 放入线程池，
          避免阻塞 asyncio 主事件循环
        - 超时等待 + continue，避免空转浪费 CPU
        - 检测客户端断连，及时退出
        - 心跳保活，防止代理服务器（如 Nginx）因无数据而断开连接
        - finally 确保清理队列资源

    Args:
        task_id:   任务唯一标识
        request:   FastAPI Request 对象，用于检测客户端断连
        heartbeat: 心跳间隔（秒），默认 30 秒。设为 0 则关闭心跳
        timeout:   queue.get 阻塞超时（秒），默认 1.0 秒

    Yields:
        标准 SSE 格式的字符串
    """
    # 1. 获取或创建队列
    stream_queue = get_or_create_sse_queue(task_id)

    # 2. 获取当前运行的异步事件循环（用于将同步阻塞操作放入线程池执行）
    loop = asyncio.get_running_loop()

    # 3. 心跳计时器（用于判断距离上次发送数据过去了多久）
    last_send_time = loop.time()

    try:
        # 4. 发送连接建立信号，告诉前端"管道已通"
        yield _sse_pack(SSEEvent.READY, {"task_id": task_id})

        while True:
            # 5. 若客户端断开，尽快退出
            if await request.is_disconnected():
                break

            # 6. 使用 run_in_executor 将同步阻塞的 queue.get 放入线程池执行
            #    block=True:   队列为空时阻塞等待
            #    timeout=1.0:  超时后抛出 queue.Empty
            try:
                msg = await loop.run_in_executor(
                    None,
                    stream_queue.get,
                    True,     # block
                    timeout,  # timeout
                )
            except queue.Empty:
                # 7. 如果队列为空（超时），检查心跳
                if heartbeat > 0:
                    now = loop.time()
                    if now - last_send_time >= heartbeat:
                        # 发送心跳包，保持连接活跃
                        yield _sse_pack(SSEEvent.PING, "")
                        last_send_time = now
                        continue

                # 继续监听
                continue

            # 8. 解析队列中获取到的消息体
            event = msg.get("event")
            data = msg.get("data")

            # 9. 如果是 DONE 事件，发送完就退出
            if event == SSEEvent.DONE:
                yield _sse_pack(event, data)
                break

            # 10. 将正常的事件和数据打包成标准 SSE 格式，通过 yield 推送给前端
            yield _sse_pack(event, data)

            # 11. 更新最后发送时间（心跳依据）
            last_send_time = loop.time()

    except (ConnectionResetError, BrokenPipeError):
        # 客户端强行刷新页面或关闭标签页，TCP 管道破裂，静默退出
        return
    except asyncio.CancelledError:
        # 协程被取消：重新抛出，让外层框架知道它被成功取消了
        raise
    finally:
        # 清理资源，防止内存泄漏
        remove_sse_queue(task_id)


# ============================================================
# 5. 便捷工厂函数
# ============================================================

def create_sse_response(task_id: str, request: Request, **kwargs) -> StreamingResponse:
    """
    创建一个配置好的 StreamingResponse，可直接作为 FastAPI 路由的返回值。

    Usage::

        @app.get("/stream/{task_id}")
        async def stream(task_id: str, request: Request):
            return create_sse_response(task_id, request)

    Args:
        task_id: 任务唯一标识
        request: FastAPI Request 对象
        **kwargs: 传递给 sse_generator 的额外参数（如 heartbeat, timeout）

    Returns:
        配置好 headers 的 StreamingResponse 实例
    """
    return StreamingResponse(
        content=sse_generator(task_id, request, **kwargs),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 禁用 Nginx 缓冲
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


# ============================================================
# 6. 辅助函数：发送结束信号
# ============================================================

def signal_done(task_id: str, data: Any = None) -> None:
    """
    向指定任务的 SSE 流发送结束信号。

    业务代码处理完成后调用此函数，告知前端数据流已结束。
    发送后 sse_generator 会自动退出循环并清理队列资源。

    Args:
        task_id: 任务唯一标识
        data:    可选的结束附带数据（如最终结果摘要）
    """
    push_sse_message(task_id, SSEEvent.DONE, data or {})


def signal_error(task_id: str, error_message: str) -> None:
    """
    向指定任务的 SSE 流发送错误信号。

    Args:
        task_id:       任务唯一标识
        error_message: 错误描述信息
    """
    push_sse_message(task_id, SSEEvent.ERROR, {"message": error_message})