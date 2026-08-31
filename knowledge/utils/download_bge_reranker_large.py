
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3")
model.save("D:\\$AAA\\0.software\\0、Large Models\\models")
print("bge-reranker-v2-m3 模型下载完成")

