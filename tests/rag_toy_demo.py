
"""
流程：文档加载 → 文本分块 → 向量化入库 → 检索召回 → 拼接 Prompt → 调用 LLM 生成答案
"""


import os
import chromadb
from dashscope import TextEmbedding, Generation

import dashscope
from dotenv import load_dotenv

# 读取 .env 文件
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def chunk_text(text: str, chunk_size: int = 500):
    """
        将文本分块，每块大小为 chunk_size
    Args:
        text (str): 文本内容
        chunk_size (int): 每块大小
    Returns:  文本块列表
    """
    chunks = []  # 存储分块后的文本
    start = 0 # 启始位置
    # 循环，直到文本被完全分块
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size
    return chunks     

def get_embeddings(text: str):
    resp = TextEmbedding.call(
        model="text-embedding-v3",
        input=text,
        dimension=1024
    )

    if resp.status_code != 200:
        raise ValueError(f"获取向量接口失败! {resp.message}")
     # 提取向量
    return resp.output['embeddings'][0]['embedding']

def llm_answer(question: str, context: str):
    """
        llm生成答案
    Args:
        question (str): 用户问题
        context (str): 文档内容
    Returns:  答案
    """
    prompt = f"""
你是电商知识库问答助手，**严格只使用下面提供的参考文档内容回答问题**。
如果参考文档没有答案，直接回答：【知识库中未找到相关信息】，不要编造内容。

=====参考文档=====
{context}
=====用户问题=====
{question}
"""
    resp = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        result_format="message",
    )
    if resp.status_code != 200:
        raise ValueError(f"大模型调用失败! {resp.message}")
    return resp.output.choices[0].message.content

if __name__ == '__main__':
    # 模拟5份txt文档
    raw_docs = [
        "退货政策：7天无理由退货，商品需保持完好不影响二次销售。生鲜品类不支持7天无理由。7天从签收当日起算。",
        "退款流程：用户提交退款申请后，平台审核，审核通过后1~3个工作日原路退款。银行卡到账会慢一些。",
        "会员规则：会员分为金、银、铜三级。等级根据近30天消费金额自动计算。会员可享受专属优惠券。优惠券不可叠加。",
        "客服时间：人工客服工作日9:00~18:00，周末无人工客服。紧急问题可提交工单。",
        "生鲜售后：生鲜签收后24小时内反馈坏损，按坏损比例赔付。超时反馈不予赔付。"
    ]
    # 文档切分
    all_chunks = []
    for doc in raw_docs:
        chunks = chunk_text(doc, 2000)
        all_chunks.extend(chunks)
    print(f"文档切分完成，共{len(all_chunks)}个chunk")

    # 向量化入库
    chroma_client = chromadb.PersistentClient(path="./chromadb_data")
    # 创建或获取集合
    coll = chroma_client.get_or_create_collection(name="rag_toy_collection")
    # 将切片的文档向量化
    emb_list = [get_embeddings(chunk) for chunk in all_chunks]
    # 将向量化的文档存入集合
    coll.add(
        documents=all_chunks,  # 文档内容
        embeddings=emb_list,   # 文档向量
        ids=[f"chunk_{i}" for i in range(len(all_chunks))]  # 文档id
    )
    print("文档向量化入库完成！")
    # 用户提问
    user_query = "充会员有什么好处啊?"
    print(f"用户提问：{user_query}")    
    # 用户提问转为向量
    user_emb = get_embeddings(user_query)
    res = coll.query(query_embeddings=[user_emb], n_results=3)
    retrieved_chunks = res["documents"][0]
    print(f"召回的3个chunk：{retrieved_chunks}")
    # 把召回的3块文本拼接成上下文context
    context = "\n".join(retrieved_chunks)
    print(f"\n召回的参考上下文：\n{context}")

    # 送入大模型生成答案
    answer = llm_answer(user_query, context)
    print(f"大模型生成的答案：\n{answer}")

