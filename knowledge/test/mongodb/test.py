from pymongo import MongoClient

#创建mongodb客户端对象
mongodb_client=MongoClient("mongodb://218.244.155.46:27017")

# 创建（选择）数据库
db = mongodb_client["cailusheng"]

# 创建（选择）集合
collection = db["students"]

def create_collection():

    # result=collection.insert_one(
    #     {
    #         "name": "张三",
    #         "age": 20,
    #         "major": "计算机科学"
    #     }
    # )
    result = collection.insert_many(
        [
            {"name": "李四", "age": 22, "major": "软件工程"},
            {"name": "王五", "age": 21, "major": "计算机科学"},
        ]
    )
    print(result)

def find_collection():
    for doc in collection.find({'name': '王五'}):
        print(doc['name'])

def find_max_age():
    for doc in collection.find().sort("age",-1).limit(2):
        print(doc)

def update_one():
    result=collection.update_one(
        {"name":"张三"},
        {"$set":{"age":100}}
    )
    print(result)

if __name__=="__main__":
    # find_collection()
    # find_max_age()
    update_one()