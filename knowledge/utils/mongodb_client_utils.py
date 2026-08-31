import os
from datetime import datetime
from typing import List

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

class MongoDbClient:
    def __init__(self):
        #获取mongodb客户端对象
        self.client=MongoClient(os.getenv("MONGO_URL"))
        #获取数据库
        self.db=self.client["MONGO_DB_NAME"]
        #获取集合
        self.chat_message=self.db["chat_message"]

def get_mongodb_client():
    return MongoDbClient()


def get_recent_message(session_id:str,limit:int=10):
    """查询前十条对话"""
    #获取客户端对象
    mongodb_client=get_mongodb_client()

    #根据session_id查询最近（倒序排序）前十条信息
    result=mongodb_client.chat_message.find({"session_id":session_id}).sort("ts",-1).limit(limit)

    #返回列表结果
    return list(result)


def save_chat_message(
    session_id:str,
    role:str,
    text:str,
    rewritten_query:str="",
    item_names:List[str]=None,
    message_id:str=None)->str:
    """保存或者更新对话信息"""

    #获取客户端对象
    mongodb_client=get_mongodb_client()

    #获取时间戳
    ts=datetime.now().timestamp()

    #构建添加数据
    data={
        "session_id":session_id,
        "role":role,
        "text":text,
        "rewritten_query":rewritten_query,
        "item_names":item_names,
        "ts":ts,
    }
    if message_id:
        mongodb_client.chat_message.update_one(
            {"_id":ObjectId(message_id)},
            {"$set":data}
        )
        return message_id
    else:
        result=mongodb_client.chat_message.insert_one(data)
        return str(result.inserted_id)

def delete_chat_message(session_id:str):
    """清空session_id下的所有对话记录"""
    mongodb_client=get_mongodb_client()
    result=mongodb_client.chat_message.delete_many({"session_id":session_id})
    return str(result.deleted_count)





