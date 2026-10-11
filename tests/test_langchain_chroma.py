

import os


from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings


load_dotenv()
# 配置 Embeddings 模型 (以Qwen为例)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

# 2. LangChain封装的Chroma向量存储
# persist_directory：持久化存储目录
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="test_collection"
)

# 要添加文档
# 3. 构造Document对象（LangChain标准文档对象，带page_content + metadata）
docs = [
    Document(page_content="退货政策是7天无理由", metadata={"source": "售后手册.md"}),
    Document(page_content="退款流程需要3个工作日", metadata={"source": "售后手册.md"}),
    Document(page_content="发票开具请联系财务", metadata={"source": "平台规则.docx"}),
    Document(page_content="账号密码重置在设置页面", metadata={"source": "用户手册.md"}),
    Document(page_content="会员等级分为金银铜三级", metadata={"source": "会员规则.md"}),
    Document(page_content="优惠券不可叠加使用", metadata={"source": "活动规则.pdf"}),
    Document(page_content="生鲜商品不支持无理由退货", metadata={"source": "生鲜售后.docx"}),
    Document(page_content="配送范围仅限中国大陆", metadata={"source": "配送说明.md"}),
    Document(page_content="客服工作时间为9-18点", metadata={"source": "客服规范.md"}),
    Document(page_content="今天的天气真不错啊", metadata={"source": "闲聊文本"})
]


# 添加向量
vectorstore.add_documents(docs)
# 查询：先把query转向量再检索
query_text = "退款流程需要多久"

# 语义查询, 查询文本，内部自动调用embedding再检索
results = vectorstore.similarity_search_with_score(query_text, k=3)

# 6. 查看结果
print("=== LangChain 查询结果 ===")
for doc, score in results:
    print(f"内容: {doc.page_content}，距离: {score:.4f}")
