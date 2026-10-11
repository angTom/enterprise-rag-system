import os

import dashscope


from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
# input_texts = "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买"
input_texts = ["退货政策","退款流程","今天的天气"]

load_dotenv()

# 基于 dashscope.SDK 实例化
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
resp = dashscope.TextEmbedding.call(
model="text-embedding-v3",
input=input_texts,
dimension=1024
)


# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://llm-5hziie3itqk1m81r.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    
# )
# resp = client.embeddings.create(
#     model="qwen3.7-text-embedding-flash",
#     input=input_texts,
#     dimensions=1024, # 指定向量维度
#     encoding_format="float", # 指定输入格式
# )
# print(resp.model_dump_json())

if resp.status_code != 200:
    raise ValueError("API 返回了空的 embeddings 列表")

# 提取向量列表
vectors = [item['embedding'] for item in resp.output['embeddings']]
# 3. 打印向量看一眼（直观感受高维浮点数组）
print("=" * 50)
print("【向量观察】'退货政策' 的前10个维度:")
print(vectors[0][:10])
print(f"... (共 {len(vectors[0])} 维)")
print("=" * 50)

# 计算余弦相似度

def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

sim12 = cosine_similarity(vectors[0], vectors[1])
sim13 = cosine_similarity(vectors[0], vectors[2])

print(f"\n【距离验证】")
print(f"'退货政策' ↔ '退款流程' 的距离: {sim12:.4f}")
print(f"'退货政策' ↔ '今天的天气' 的距离: {sim13:.4f}")
print(f"\n✅ 验证结果: {sim12 < sim13}")
print("   → 语义相近的文本值越接近1，向量距离越近！")