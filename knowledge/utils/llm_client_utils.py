import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from knowledge.processor.import_process.exceptions import LLMError

load_dotenv()
loger=logging.getLogger(__name__)

def get_llm_client(response_format:bool=False):

    model_kwargs={}
    if response_format:
        model_kwargs["response_format"]={"type":"json_object"}

    try:
        llm_client=ChatOpenAI(
            model=os.getenv("LLM_DEFAULT_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.1,
            extra_body={"enable_thinking":False},
            #强制llm必须以json格式返回
            model_kwargs=model_kwargs
        )
        return llm_client
    except LLMError as e:
        loger.exception(f"大模型客户端初始化失败：{e}")
        raise
