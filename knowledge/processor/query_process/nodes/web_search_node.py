import asyncio
import json
import os
from typing import Dict, Any

from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState

load_dotenv()

class WebSearchNode(BaseNode):
    name = "web_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        #参数校验，获取前一个节点的rewritten_query
        rewritten_query=state.get("rewritten_query")
        if not rewritten_query:
            raise StateFieldError(node_name=self.name,field_name="rewritten_query")

        #异步调用MCP工具，获取结果
        mcp_result=asyncio.run(self.call_mcp_execute_web_search(rewritten_query=rewritten_query))

        if not mcp_result:
            return {}

        return {"web_search_docs":mcp_result}

    async def call_mcp_execute_web_search(self,rewritten_query:str):

        #建立连接需要的header信息
        headers={
            "Authorization": f"Bearer {self.config.mcp_dashscope_api_key}",
            "Content-Type":"application/json"
        }


        try:
            # 创建连接对象
            mcp_client=MCPServerStreamableHttp(
                params={
                    "url":self.config.mcp_dashscope_base_url,
                    "headers":headers
                },
                cache_tools_list=True,
                name="阿里云百炼平台-MCP-联网搜索"
            )
            #建立连接
            await mcp_client.connect()

            #调用mcp服务器工具
            tool_result=await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query":rewritten_query,
                    "count":5
                }
            )
            if not tool_result:
                return []

            #处理MCP服务器工具返回的结果,反序列化
            content_text=tool_result.content[0].text
            data:Dict[str,Any]=json.loads(content_text)

            #收集指定内容并且返回
            mcp_result=[]
            pages=data.get("pages")
            for page in pages:
                p={
                    "snippet":page.get("snippet"),
                    "title":page.get("title"),
                    "url":page.get("url")
                }
                mcp_result.append(p)

            return mcp_result

        except Exception as e:
            raise e
        finally:
            if mcp_client:
                await mcp_client.cleanup()



if __name__=="__main__":
    state = {
        "item_names": "室内无线网关",
        "rewritten_query": "今天（2026.7.31）赣州天气怎么样"
    }
    WebSearchNode=WebSearchNode()
    result=WebSearchNode(state)
    print(result)

